// Copyright 2025 Keita HAGIWARA. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sigrpc-runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <signal.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <immintrin.h>
#include <sys/auxv.h>
#include <sys/resource.h>
#include <fcntl.h>
#ifndef HWCAP2_FSGSBASE
#define HWCAP2_FSGSBASE (1 << 1)
#endif //(HWCAP2_FSGSBASE)

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif //(_DEFAULT_SOURCE)
#include <unistd.h>
#include <shared_mutex>
#include <cassert>
#include <algorithm>

using namespace std;

alignas(16) static thread_local struct _libc_fpstate g_fpstate;
static thread_local greg_t g_greg[__NGREG];

typedef struct request_local_data_s
{
    int fd;
    InvokeFuncMsg *request_msg;
    IsolatedRuntime *rt_object;
    std::string *client_id;
    std::mutex mtx;
} rld_t;

static thread_local rld_t request_info;

static thread_local void *curr_page;

static jmp_buf ret_buf;

static void segv_handler(int signal, siginfo_t *si, void *arg)
{
    const int page_size = getpagesize();
    ucontext_t *ctx = static_cast<ucontext_t *>(arg);
    void *si_addr_page_start =
        reinterpret_cast<void *>(
            reinterpret_cast<uint64_t>(si->si_addr) & RUNTIME_H_PAGE_MASK);
    if (!si_addr_page_start)
    {
        LOG_LOCATION("invalid address");
        /* cannot recover SEGV */
        std::exit(EXIT_FAILURE);
    }
    if (curr_page == si_addr_page_start)
        longjmp(ret_buf, 1);
    { /* RAII scope, inline assembly part cannot call destructor */
        InvokeFuncMsg msg;
        rld_t *parent_tls_data;
        unsigned long long fs_base, gs_base;

        /* gs_base value is a copy of parent thread's fs_base */
        /* get current fsbase */
        fs_base = _readfsbase_u64();
        /* get current gsbase */
        gs_base = _readgsbase_u64();
        /* switch to parent tls */
        _writefsbase_u64(gs_base);
        parent_tls_data = &request_info;
        /* switch to own tls */
        _writefsbase_u64(fs_base);
        /* copy parent thread's tls variable */
        /* read only object, mutex not required */
        std::memcpy(&request_info, parent_tls_data, sizeof(request_info));

        try
        {
            int fd = request_info.fd;
            int rc;

            std::lock_guard<std::mutex> lock(request_info.rt_object->GetFDMutex());
            /* read only object, mutex not required */
            msg.CopyFrom(*request_info.request_msg);

            msg.mutable_header()->set_msg_type(INVOKEFUNC);
            msg.mutable_header()->set_status(StatusCode::NOT_FOUND);
            msg.clear_page();

            /* add request page info */
            x64::Page page;
            uint64_t want_page_start = reinterpret_cast<uint64_t>(si->si_addr) & RUNTIME_H_PAGE_MASK;
            page.set_address(want_page_start);
            page.set_runtime_revision(0);
            page.set_client_revision(0);
            page.set_content_size(0);
            msg.add_page()->CopyFrom(page);
            if (!request_info.rt_object->page_manager.FindPage(want_page_start))
            {
                request_info.rt_object->WriteMsg(fd, &msg);
                if (read(fd, &rc, sizeof(rc)) != sizeof(rc)) // remove request code
                {
                    LOG_LOCATION("read failed");
                    throw std::runtime_error(msg.header().client_id() + ": read");
                }
                std::string byte_msg = request_info.rt_object->ReadMsg(fd);
                msg.Clear();
                if (!msg.ParseFromString(byte_msg))
                {
                    LOG_LOCATION("ParseFromString failed");
                    /* cannot recover SEGV */
                    std::exit(EXIT_FAILURE);
                }
                if (msg.page().size() == 0)
                {
                    LOG_LOCATION("cannot recover from SEGV");
                    throw std::runtime_error("cannot recover from SEGV");
                }
                parent_tls_data->rt_object->page_manager.FlushClientChange(msg.page(), 0);
            }
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            /* cannot recover SEGV */
            std::exit(EXIT_FAILURE);
        }
    }
}

const std::map<StatusCode, std::string> IsolatedRuntime::StatusDescription = {
    {StatusCode::OK, "no error"},
    {StatusCode::NOT_FOUND, "invalid program counter, page fault or no runtime"},
    {StatusCode::INVALID_ARGUMENT, "invalid argument"},
    {StatusCode::INTERNAL, "system error or address space collision"},
    {StatusCode::DATA_LOSS, "broken message"},
};

std::vector<x64::Page> PageManager::get_page_diff(
    const void *new_page, void *original_page, size_t size)
{
    typedef enum
    {
        UNCHANGED,
        CHANGED
    } check_state;

    check_state state = UNCHANGED;
    std::vector<Page> diff;
    size_t change_start;
    uint8_t *new_page_copy = reinterpret_cast<uint8_t *>(alloca(size));
    if (!setjmp(ret_buf))
        std::memcpy(new_page_copy, new_page, size);
    else
    {
        curr_page = MAP_FAILED;
        return std::vector<x64::Page>();
    }
    const uint8_t *new_byte_ptr = static_cast<const uint8_t *>(new_page_copy);
    uint8_t *original_byte_ptr = static_cast<uint8_t *>(original_page);

    size_t offset = 0;
    while (offset < size)
    {
        if (*new_byte_ptr != *original_byte_ptr && state == UNCHANGED)
        {
            state = CHANGED;
            change_start = offset;
        }

        if (*new_byte_ptr == *original_byte_ptr && state == CHANGED)
        {
            state = UNCHANGED;
            Page p;
            p.set_address(reinterpret_cast<uint64_t>(new_page) + change_start);
            p.set_content_size(offset - change_start);
            p.set_allocated_content(new std::string(
                reinterpret_cast<char *>(new_page_copy) + change_start,
                p.content_size()));
            diff.push_back(p);
        }

        ++new_byte_ptr;
        ++original_byte_ptr;
        ++offset;
    }

    if (state == CHANGED)
    {
        Page p;
        p.set_address(reinterpret_cast<uint64_t>(new_page) + change_start);
        p.set_content_size(offset - change_start);
        p.set_allocated_content(new std::string(
            reinterpret_cast<char *>(new_page_copy) + change_start,
            p.content_size()));
        diff.push_back(p);
    }
    std::memcpy(original_page, new_page_copy, size);

    return diff;
}

void PageManager::FlushClientChange(const google::protobuf::RepeatedPtrField<x64::Page> src, uint64_t stack_bottom)
{
    std::set<struct mem_region_s> skip_page_list;
    uint64_t page_start = 0;
    uint64_t next_client_revision = 0;
    std::unique_lock<std::mutex> lock(map_mtx);
    for (x64::Page p : src)
    {
        if (skip_page_list.find({.start = p.address(), .end = p.address() + p.content_size()}) != skip_page_list.end())
            continue;
        if (page_start && (p.address() & RUNTIME_H_PAGE_MASK) != page_start)
            try
            {
                update_client_revision(page_start, next_client_revision);
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error(e.what());
            }
        if (p.content_size() && (p.address() & RUNTIME_H_PAGE_MASK) != page_start)
        {
            page_start = p.address() & RUNTIME_H_PAGE_MASK;
            next_client_revision = p.client_revision();
        }
        try
        {
            flush_client_page_change(p, stack_bottom, skip_page_list, lock);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(e.what());
        }
    }

    if (src.size() && page_start)
        try
        {
            update_client_revision(page_start, next_client_revision);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(e.what());
        }
}

void PageManager::update_client_revision(uint64_t page_start, uint64_t next_client_revision)
{
    if (page_state.find(page_start) == page_state.end())
    {
        LOG_LOCATION("tried to update invalid page revision");
        throw std::runtime_error("tried to update invalid page revision");
    }
    page_state[page_start].page->set_client_revision(next_client_revision);

    /* wake up successor revision thread */
    if (page_state.at(page_start).successor_list.find(next_client_revision + 1) !=
        page_state.at(page_start).successor_list.end())
    {
        page_state[page_start].successor_list.at(next_client_revision + 1)->notified = true;
        page_state[page_start].successor_list.at(next_client_revision + 1)->cv.notify_one();
    }
}

bool PageManager::in_mem_region(PageManager::mem_region_s mem_region, uint64_t addr)
{
    return mem_region.start <= addr && addr < mem_region.end;
}

bool PageManager::in_stack_region(x64::Page page)
{
    std::set<struct mem_region_s>::iterator stack_page_iter = stack_page_tbl.upper_bound({
        .start = page.address(),
    });
    if (stack_page_iter != stack_page_tbl.begin())
        stack_page_iter--;
    if (stack_page_iter == stack_page_tbl.end())
        return false;
    return in_mem_region(*stack_page_iter, page.address());
}

void PageManager::wait_prev_revision(x64::Page page, std::unique_lock<std::mutex> &lock)
{
    if (page_state.at(page.address() & RUNTIME_H_PAGE_MASK).page_type == TYPE_WAITMMAP ||
        page.client_revision() > page_state.at(page.address() & RUNTIME_H_PAGE_MASK).page->client_revision() + 1)
    {
        std::shared_ptr<struct successor_revision_s> entry = make_shared<struct successor_revision_s>();
        entry->notified = false;
        page_state[page.address() & RUNTIME_H_PAGE_MASK].successor_list[page.client_revision()] = entry;
        entry->cv.wait(lock, [entry]
                       { return entry->notified; });
        page_state[page.address() & RUNTIME_H_PAGE_MASK].successor_list.erase(page.client_revision());
    }
}

void PageManager::flush_client_stack(x64::Page src, uint64_t stack_bottom, std::set<struct mem_region_s> &skip_page_list, std::unique_lock<std::mutex> &lock)
{
    if (src.client_revision() == 0)
    {
        std::set<struct mem_region_s>::iterator stack_page_iter = stack_page_tbl.upper_bound({
            .start = src.address(),
        });
        if (stack_page_iter != stack_page_tbl.begin())
            stack_page_iter--;

        if (stack_page_iter == stack_page_tbl.end() || !in_mem_region(*stack_page_iter, src.address()))
        {
            RAIIFD raii_fd = RAIIFD(memfd_create("", MFD_CLOEXEC));
            if (raii_fd.fd == -1)
            {
                LOG_LOCATION("memfd_create failed");
                throw std::runtime_error(client_id + ": memfd_create");
            }
            struct rlimit limit;
            if (getrlimit(RLIMIT_STACK, &limit) == -1)
            {
                LOG_LOCATION("getrlimit failed");
                throw std::runtime_error(client_id + ": getrlimit");
            }
            assert(limit.rlim_cur > 0x100000);
            struct mem_region_s mem_region;
            uint64_t stack_size = limit.rlim_cur - 0x100000;
            if (ftruncate(raii_fd.fd, stack_size) == -1)
            {
                LOG_LOCATION("ftruncate failed");
                throw std::runtime_error(client_id + ": ftruncate failed");
            }
            uint64_t map_addr = stack_bottom - stack_size;
            mem_region.start = map_addr;
            mem_region.end = map_addr + stack_size;

            void *wr_start = mmap(nullptr, mem_region.end - mem_region.start,
                                  PROT_WRITE | PROT_READ, MAP_SHARED, raii_fd.fd, 0);
            if (wr_start == MAP_FAILED)
            {
                LOG_LOCATION("mmap failed");
                throw std::runtime_error(client_id + ": mmap");
            }
            mem_region.wr_start = reinterpret_cast<uint64_t>(wr_start);

            void *bias = reinterpret_cast<void *>(map_addr);
            void *mapped_addr = mmap(bias, stack_size, PROT_NONE,
                                     MAP_SHARED, raii_fd.fd, 0);
            if (mapped_addr != bias)
            {
                LOG_LOCATION("mmap stack failed at " << bias << ", src.address=" << reinterpret_cast<void *>(src.address()) << ", content_size=" << reinterpret_cast<void *>(src.content_size()) << ", stack_bottom=" << reinterpret_cast<void *>(stack_bottom));
                if (mapped_addr != MAP_FAILED)
                    munmap(mapped_addr, stack_size);
                munmap(wr_start, stack_size);
                throw std::runtime_error(client_id + ": mmap stack failed");
            }
            if (page_state.find(src.address()) == page_state.end())
            {
                page_state[src.address()] = {
                    .page = make_shared<x64::Page>(src),
                    .page_type = TYPE_PULLPAGE,
                };
            }
            else
            {
                page_state[src.address()].page = make_shared<x64::Page>(src);
                page_state[src.address()].page_type = TYPE_PULLPAGE;
            }
            uint64_t offset = src.address() - map_addr;
            std::memcpy(reinterpret_cast<void *>(mem_region.wr_start + offset), src.content().data(), src.content_size());
            std::memcpy(page_state.at(src.address()).page->mutable_content()->data(), src.content().data(), src.content_size());
            mem_region.sync_region = make_shared<std::set<struct mem_region_s>>();
            mem_region.sync_region->insert({
                .start = src.address(),
                .end = src.address() + src.content_size(),
            });
            stack_page_tbl.insert(mem_region);
            uint64_t mprotect_size = src.address() + src.content_size() - mem_region.start;
            if (mprotect(bias, mprotect_size, PROT_WRITE | PROT_READ) == -1)
            {
                LOG_LOCATION("mprotect failed");
                munmap(wr_start, mem_region.end - mem_region.start);
                munmap(bias, mem_region.end - mem_region.start);
                throw std::runtime_error(client_id + ": mprotect failed");
            }
            return;
        }
        else
        {
            if (page_state.find(src.address()) == page_state.end())
            {
                page_state[src.address()] = {
                    .page = make_shared<x64::Page>(src),
                    .page_type = TYPE_PULLPAGE,
                };
            }
            else
            {
                page_state[src.address()].page = make_shared<x64::Page>(src);
                page_state[src.address()].page_type = TYPE_PULLPAGE;
            }
            uint64_t offset = src.address() - stack_page_iter->start;
            void *wr_addr = reinterpret_cast<void *>(stack_page_iter->wr_start + offset);
            std::memcpy(wr_addr, src.content().data(), src.content_size());
            std::memcpy(page_state.at(src.address()).page->mutable_content()->data(), src.content().data(), src.content_size());
            if (mprotect(reinterpret_cast<void *>(src.address()), src.content_size(), PROT_WRITE | PROT_READ) == -1)
            {
                LOG_LOCATION("mprotect failed");
                throw std::runtime_error(client_id + ": mprotect failed");
            }
            std::set<struct mem_region_s>::iterator sync_region_iter = stack_page_iter->sync_region->upper_bound({
                .start = src.address(),
            });
            enum update_region_action_t
            {
                NEW_REGION,
                CONCAT_PREV,
                CONCAT_NEXT,
                CONCAT_PREV_AND_NEXT,
            };
            update_region_action_t act = NEW_REGION;
            if (sync_region_iter != stack_page_iter->sync_region->end())
            {
                if (src.address() + src.content_size() == sync_region_iter->start)
                    act = CONCAT_NEXT;
            }
            if (sync_region_iter != stack_page_iter->sync_region->begin())
                sync_region_iter--;
            assert(sync_region_iter != stack_page_iter->sync_region->end());
            if (src.address() + src.content_size() == sync_region_iter->start)
            {
                if (act == CONCAT_NEXT)
                    act = CONCAT_PREV_AND_NEXT;
                else
                    act = CONCAT_PREV;
            }
            switch (act)
            {
            case CONCAT_PREV_AND_NEXT:
            {
                struct mem_region_s prev_region = *sync_region_iter++;
                struct mem_region_s next_region = *sync_region_iter;
                stack_page_iter->sync_region->erase(prev_region);
                stack_page_iter->sync_region->erase(next_region);
                struct mem_region_s new_region = {.start = prev_region.start, .end = next_region.end};
                stack_page_iter->sync_region->insert(new_region);
                break;
            }
            case CONCAT_PREV:
            {
                struct mem_region_s prev_region = *sync_region_iter;
                stack_page_iter->sync_region->erase(prev_region);
                struct mem_region_s new_region = {.start = prev_region.start, .end = src.address() + src.content_size()};
                stack_page_iter->sync_region->insert(new_region);
                break;
            }
            case CONCAT_NEXT:
            {
                struct mem_region_s next_region = *(++sync_region_iter);
                stack_page_iter->sync_region->erase(next_region);
                struct mem_region_s new_region = {.start = src.address(), .end = next_region.end};
                stack_page_iter->sync_region->insert(new_region);
                break;
            }
            case NEW_REGION:
            {
                struct mem_region_s new_region = {
                    .start = src.address(),
                    .end = src.address() + src.content_size(),
                };
                stack_page_iter->sync_region->insert(new_region);
                break;
            }
            }
            return;
        }
    }
    if (src.content_size() == 0)
    {
        page_state.erase(src.address());
        std::set<struct mem_region_s>::iterator stack_page_iter = stack_page_tbl.upper_bound({
            .start = src.address(),
        });
        if (stack_page_iter != stack_page_tbl.begin())
            stack_page_iter--;
        if (stack_page_iter == stack_page_tbl.end())
            return;
        if (!in_mem_region(*stack_page_iter, src.address()))
            return;
        for (struct mem_region_s region : *stack_page_iter->sync_region)
            skip_page_list.insert(region);
        munmap(reinterpret_cast<void *>(stack_page_iter->start), stack_page_iter->end - stack_page_iter->start);
        stack_page_tbl.erase(stack_page_iter);
        return;
    }
    std::set<struct mem_region_s>::iterator stack_page_iter = stack_page_tbl.upper_bound({
        .start = src.address(),
    });
    if (stack_page_iter != stack_page_tbl.begin())
        stack_page_iter--;
    assert(in_mem_region(*stack_page_iter, src.address()));
    uint64_t offset = src.address() - stack_page_iter->start;
    std::memcpy(reinterpret_cast<void *>(stack_page_iter->wr_start + offset), src.content().data(), src.content_size());
    offset = src.address() & (~RUNTIME_H_PAGE_MASK);
    std::memcpy(page_state.at(src.address() & RUNTIME_H_PAGE_MASK).page->mutable_content()->data() + offset, src.content().data(), src.content_size());
}

void PageManager::flush_client_heap(x64::Page src, std::unique_lock<std::mutex> &lock)
{
    if (src.client_revision() == 0)
    {
        RAIIFD raii_fd = RAIIFD(memfd_create("", MFD_CLOEXEC));
        if (raii_fd.fd == -1)
        {
            LOG_LOCATION("memfd_create failed");
            throw std::runtime_error(client_id + ": memfd_create");
        }
        if (ftruncate(raii_fd.fd, src.content_size()) == -1)
        {
            LOG_LOCATION("ftruncate failed");
            throw std::runtime_error(client_id + ": ftruncate failed");
        }
        struct mem_region_s mem_region;
        mem_region.start = src.address();
        mem_region.end = src.address() + src.content_size();

        void *wr_start = mmap(nullptr, src.content_size(),
                              PROT_WRITE | PROT_READ, MAP_SHARED, raii_fd.fd, 0);
        if (wr_start == MAP_FAILED)
        {
            LOG_LOCATION("mmap failed");
            throw std::runtime_error(client_id + ": mmap");
        }
        mem_region.wr_start = reinterpret_cast<uint64_t>(wr_start);

        void *bias = reinterpret_cast<void *>(src.address());
        void *mapped_addr = mmap(bias, src.content_size(), PROT_NONE,
                                 MAP_SHARED, raii_fd.fd, 0);
        if (mapped_addr != bias)
        {
            if (mapped_addr != MAP_FAILED)
                munmap(mapped_addr, src.content_size());
            munmap(wr_start, src.content_size());
            throw std::runtime_error(client_id + ": mmap stack failed");
        }
        if (page_state.find(src.address()) == page_state.end())
        {
            page_state[src.address()] = {
                .page = make_shared<x64::Page>(src),
                .page_type = TYPE_PULLPAGE,
            };
        }
        else
        {
            page_state[src.address()].page = make_shared<x64::Page>(src);
            page_state[src.address()].page_type = TYPE_PULLPAGE;
        }
        std::memcpy(wr_start, src.content().data(), src.content_size());
        std::memcpy(page_state.at(src.address()).page->mutable_content()->data(), src.content().data(), src.content_size());
        heap_page_tbl.insert(mem_region);
        uint64_t mprotect_size = src.address() + src.content_size() - mem_region.start;
        if (mprotect(bias, mprotect_size, PROT_WRITE | PROT_READ | PROT_EXEC) == -1)
        {
            LOG_LOCATION("mprotect failed");
            munmap(wr_start, mem_region.end - mem_region.start);
            munmap(bias, mem_region.end - mem_region.start);
            throw std::runtime_error(client_id + ": mprotect failed");
        }
        return;
    }
    if (src.content_size() == 0)
    {
        page_state.erase(src.address());
        std::set<struct mem_region_s>::iterator heap_page_iter = heap_page_tbl.upper_bound({
            .start = src.address(),
        });
        munmap(reinterpret_cast<void *>(heap_page_iter->start), heap_page_iter->end - heap_page_iter->start);
        heap_page_tbl.erase(heap_page_iter);
        return;
    }
    std::set<struct mem_region_s>::iterator heap_page_iter = heap_page_tbl.upper_bound({
        .start = src.address() & RUNTIME_H_PAGE_MASK,
    });
    if (heap_page_iter != heap_page_tbl.begin())
        heap_page_iter--;
    assert(in_mem_region(*heap_page_iter, src.address() & RUNTIME_H_PAGE_MASK));
    uint32_t offset = src.address() & (~RUNTIME_H_PAGE_MASK);
    void *wr_addr = reinterpret_cast<void *>(heap_page_iter->wr_start + offset);
    std::memcpy(wr_addr, src.content().data(), src.content_size());
    std::memcpy(page_state.at(src.address() & RUNTIME_H_PAGE_MASK).page->mutable_content()->data() + offset, src.content().data(), src.content_size());
}

void PageManager::flush_client_page_change(x64::Page src, uint64_t stack_bottom, std::set<struct mem_region_s> &skip_page_list, std::unique_lock<std::mutex> &lock)
{
    struct rlimit limit;
    assert(!getrlimit(RLIMIT_STACK, &limit));
    if (src.client_revision() != 0 && page_state.find(src.address() & RUNTIME_H_PAGE_MASK) == page_state.end())
    {
        page_state[src.address() & RUNTIME_H_PAGE_MASK] = {
            .page = nullptr,
            .page_type = TYPE_WAITMMAP,
        };
    }
    if (src.client_revision() != 0)
        wait_prev_revision(src, lock);
    struct mem_region_s curr_stk_region = {
        .start = stack_bottom - limit.rlim_cur + 0x100000,
        .end = stack_bottom,
    };
    if (in_stack_region(src) || (stack_bottom && in_mem_region(curr_stk_region, src.address())))
    {
        try
        {
            flush_client_stack(src, stack_bottom, skip_page_list, lock);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
        return;
    }
    flush_client_heap(src, lock);
}

std::vector<x64::Page> PageManager::flush_local_page_change(std::shared_ptr<x64::Page> src)
{
    if (page_state.find(src->address()) != page_state.end() &&
        page_state.at(src->address()).page_type == TYPE_PUSHPAGE)
    {
        void *try_mmap = mmap(
            reinterpret_cast<void *>(src->address()),
            src->content_size(),
            PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (try_mmap == MAP_FAILED)
        {
            LOG_LOCATION("mmap failed");
            throw std::runtime_error(client_id + ": mmap failed");
        }
        munmap(try_mmap, src->content_size());
        if (try_mmap == reinterpret_cast<void *>(src->address()))
        {
            x64::Page release_page;
            release_page.set_address(src->address());
            release_page.set_content_size(0);
            std::vector<x64::Page> diff(1);
            diff[0] = release_page;
            return diff;
        }
    }

    curr_page = reinterpret_cast<void *>(src->address());

    std::vector<x64::Page> diff_list = get_page_diff(
        reinterpret_cast<void *>(src->address()),
        src->mutable_content()->data(),
        src->content_size());

    if (curr_page == MAP_FAILED)
    {
        assert(page_state.at(src->address()).page_type == TYPE_PUSHPAGE);
        x64::Page release_page;
        release_page.set_address(src->address());
        release_page.set_content_size(0);
        std::vector<x64::Page> diff(1);
        diff[0] = release_page;
        return diff;
    }
    if (diff_list.size() == 0)
        return diff_list;
    else
        src->set_runtime_revision(src->runtime_revision() + 1);
    for (x64::Page &diff : diff_list)
        diff.set_runtime_revision(src->runtime_revision());
    return diff_list;
}

std::vector<x64::Page> PageManager::FlushLocalChange()
{
    std::vector<x64::Page> diff_list;
    {
        std::lock_guard<std::mutex> lock(map_mtx);
        std::vector<uint64_t> keys(page_state.size());
        for (std::map<uint64_t, struct page_state_s>::iterator entry = page_state.begin(); entry != page_state.end(); entry++)
        {
            if (!entry->second.page)
                continue;
            std::vector<x64::Page> diff = flush_local_page_change(entry->second.page);
            if (diff.size())
            {
                diff_list.insert(diff_list.end(), diff.begin(), diff.end());
                if (diff.at(0).content_size() == 0)
                {
                    heap_page_tbl.erase({.start = entry->second.page->address()});
                    entry = page_state.erase(entry);
                }
            }
        }
    }
    return diff_list;
}

void PageManager::AddPushPage(x64::Page page)
{
    std::lock_guard<std::mutex> lock(map_mtx);
    page_state[page.address()] = {
        .page = make_shared<x64::Page>(page),
        .page_type = TYPE_PUSHPAGE,
    };
    heap_page_tbl.insert({
        .start = page.address(),
        .end = page.address() + getpagesize(),
        .wr_start = page.address(),
        .sync_region = nullptr,
    });
}

bool PageManager::FindPage(uint64_t page_start)
{
    std::lock_guard<std::mutex> lock(map_mtx);
    return page_state.find(page_start) != page_state.end();
}

void PageManager::WaitPrevClientFlush(uint64_t invokefunc_id)
{
    std::unique_lock<std::mutex> lock(invokefunc_mtx);
    if (invokefunc_id != runnable_invokefunc_id)
    {
        std::shared_ptr<condition_variable> cv = make_shared<std::condition_variable>();
        invokefunc_mtx_list[invokefunc_id] = cv;
        cv->wait(lock, [this, invokefunc_id]
                 { return this->runnable_invokefunc_id == invokefunc_id; });
        invokefunc_mtx_list.erase(invokefunc_id);
    }
    runnable_invokefunc_id++;
    if (invokefunc_mtx_list.find(runnable_invokefunc_id) != invokefunc_mtx_list.end())
    {
        std::shared_ptr<condition_variable> cv = invokefunc_mtx_list.at(runnable_invokefunc_id);
        cv->notify_one();
    }
}

void IsolatedRuntime::SetRuntime(int child_pid)
{
    this->fork_state = child_pid;
}

int IsolatedRuntime::WaitRuntime(int epoll_fd)
{
    struct epoll_event ev = {0};
    switch (epoll_wait(epoll_fd, &ev, 1, 100))
    {
    case -1:
        LOG_LOCATION("epoll_wait failed");
        return 0;
    case 0:
        LOG_LOCATION("never called listen, timeout");
        return 0;
    default:
        if (!(ev.events | EPOLLIN))
        {
            LOG_LOCATION("cannot detect listen");
            return 0;
        }
    }
    return 1;
}

int IsolatedRuntime::HookTerminate(int gc_epoll_fd)
{
    int sock = ConnectRuntimeSocket();
    if (sock == -1)
    {
        LOG_LOCATION("ConnectRuntimeSocket failed");
        return -1;
    }
    REQUEST_CODE rc = REQUEST_CODE::HEALTHCHECK;
    if (write(sock, &rc, sizeof(rc)) != sizeof(rc))
    {
        LOG_LOCATION("write failed");
        close(sock);
        return -1;
    }
    struct epoll_event ev = {0};
    ev.data.fd = sock;
    ev.events = EPOLLERR | EPOLLHUP;
    if (epoll_ctl(gc_epoll_fd, EPOLL_CTL_ADD, sock, &ev) == -1)
    {
        LOG_LOCATION("epoll_ctl failed");
        close(sock);
        {
            std::lock_guard<std::mutex> lock(runtime_mtx);
            state = RUNTIME_STATE::TERMINATING;
        }
        return -1;
    }
    health_check_sock = sock;
    return sock;
}

int IsolatedRuntime::notify_listen(int sync_fd)
{
    if (write(sync_fd, &sync_fd, sizeof(sync_fd)) == -1)
    {
        LOG_LOCATION("write failed");
        return -1;
    }
    return 0;
}

void IsolatedRuntime::Serve(int min_workers, int sync_fd)
{
    this->role = ROLE::RUNTIME;
    /* set SEGV handler to catch SEGV while InvokeFunc */
    {
        struct sigaction sa = {0};
        sigemptyset(&sa.sa_mask);
        sa.sa_sigaction = segv_handler;
        sa.sa_flags = SA_SIGINFO | SA_NODEFER;
        if (sigaction(SIGSEGV, &sa, NULL) == -1)
        {
            LOG_LOCATION("sigaction failed");
            throw std::runtime_error(client_id + ": sigaction");
        }
    }
    {
        struct sigaction sa = {0};
        sa.sa_handler = SIG_IGN;
        if (sigaction(SIGPIPE, &sa, NULL) == -1)
        {
            LOG_LOCATION("sigaction failed");
            throw std::runtime_error("sigaction failed");
        }
    }

    this->min_workers = min_workers;
    struct sockaddr_un endpoint = {0};
    if (std::strlen(sun_path.c_str()) >= sizeof(endpoint.sun_path))
    {
        LOG_LOCATION("socket path is too long");
        throw std::runtime_error(client_id + ": socket path is too long");
    }
    std::strcpy(endpoint.sun_path, sun_path.c_str());
    endpoint.sun_family = AF_UNIX;
    if ((listen_request_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
    {
        LOG_LOCATION("socket failed");
        throw std::runtime_error(client_id + ": socket failed");
    }
    unlink(sun_path.c_str());
    if (bind(listen_request_fd, (struct sockaddr *)&endpoint, sizeof(struct sockaddr_un)) == -1)
    {
        LOG_LOCATION("bind failed");
        close(listen_request_fd);
        throw std::runtime_error(client_id + ": bind failed");
    }
    if (listen(listen_request_fd, SOMAXCONN) == -1)
    {
        LOG_LOCATION("listen failed errno=" << errno);
        throw std::runtime_error(client_id + ": listen failed");
    }
    int thread_key = 0;
    for (int worker = 0; worker < min_workers; worker++)
    {
        std::lock_guard<std::mutex> lock(runtime_mtx);
        workers[thread_key] = std::thread(&IsolatedRuntime::handle_request, this, thread_key);
        thread_key++;
    }
    if (notify_listen(sync_fd) == -1)
    {
        LOG_LOCATION("notify_listen failed");
        throw std::runtime_error(client_id + ": notify_listen failed");
    }
    close(sync_fd);

    epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        LOG_LOCATION("epoll_create1 failed");
        throw std::runtime_error(client_id + ": epoll_create1");
    }
    struct epoll_event ev = {0};
    ev.data.fd = listen_request_fd;
    ev.events = EPOLLIN;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_request_fd, &ev))
    {
        LOG_LOCATION("epoll_ctl failed");
        close(epoll_fd);
        throw std::runtime_error(client_id + ": epoll_ctl");
    }
    int health_check_fd = -1;
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(runtime_mtx);
            if (state == RUNTIME_STATE::TERMINATING)
                break;
        }
        std::memset(&ev, 0, sizeof(ev));
        int nfds = epoll_wait(epoll_fd, &ev, 1, timeout_ms);
        switch (nfds)
        {
        case -1:
            LOG_LOCATION("epoll_wait failed");
            throw std::runtime_error(client_id + ": epoll_wait failed");
        case 0:
        {
            std::lock_guard<std::mutex> lock(runtime_mtx);
            if (!active_workers)
            {
                LOG_LOCATION("no requests, shutdown runtime");
                state = RUNTIME_STATE::TERMINATING;
                cv.notify_all();
            }
            else if (active_workers != workers.size())
                cv.notify_one(); /* scale-in workers */
            break;
        }
        default:
            int fd = accept(listen_request_fd, nullptr, nullptr);

            if (fd == -1)
            {
                LOG_LOCATION("accept failed");
                throw std::runtime_error(client_id + ":accept failed");
            }

            /* expect REQUEST_CODE::HEALTHCHECK */
            if (state == RUNTIME_STATE::CREATED)
            {
                REQUEST_CODE rc;
                health_check_fd = fd;
                switch (read(fd, &rc, sizeof(rc)))
                {
                case sizeof(rc):
                    if (rc == REQUEST_CODE::HEALTHCHECK)
                    {
                        state = RUNTIME_STATE::RUNNING;
                        break;
                    }
                    /* fallthrough */
                default:
                    LOG_LOCATION("invalid request for RUNTIME_STATE::CREATED");
                    state = RUNTIME_STATE::TERMINATING;
                }
                continue;
            }

            std::lock_guard<std::mutex> lock(runtime_mtx);
            if (active_workers == workers.size())
            {
                workers[thread_key] = std::thread(&IsolatedRuntime::handle_request, this, thread_key);
                thread_key++;
            }
            job_queue.push(fd);
            cv.notify_one();
        }
    }
    if (health_check_fd != -1)
        close(health_check_fd);
}

std::mutex &IsolatedRuntime::GetFDMutex()
{
    return atomic_fd_mtx;
}

std::string IsolatedRuntime::ReadMsg(int fd)
{
    enum read_state
    {
        READ_MSG_SIZE,
        READ_MSG_PAYLOAD,
        READ_FINISH
    };
    int _epoll_fd = epoll_create1(0);
    if (_epoll_fd == -1)
    {
        LOG_LOCATION("epoll_create1 failed");
        throw std::runtime_error(client_id + ": epoll_create1");
    }
    struct epoll_event ev = {0};
    ev.data.fd = fd;
    ev.events = EPOLLIN;
    if (epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, fd, &ev))
    {
        LOG_LOCATION("epoll_ctl failed");
        close(_epoll_fd);
        throw std::runtime_error(client_id + ": epoll_ctl");
    }
    uint64_t msg_size = 0;
    read_state rs = read_state::READ_MSG_SIZE;
    ssize_t read_total = 0;
    std::string buf;
    while (rs != read_state::READ_FINISH)
    {
        switch (epoll_wait(_epoll_fd, &ev, 1, timeout_ms))
        {
        case -1:
            LOG_LOCATION("epoll_wait failed");
            close(_epoll_fd);
            throw std::runtime_error(client_id + ": epoll_wait");
        case 0:
            LOG_LOCATION("no data");
            close(_epoll_fd);
            throw std::runtime_error(client_id + ": no data");
        default:
            if (ev.events & EPOLLIN)
                break;
            if (ev.events & (EPOLLERR | EPOLLHUP))
            {
                LOG_LOCATION("invalid fd");
                close(_epoll_fd);
                throw std::runtime_error(client_id + ": invalid fd");
            }
        }
        ssize_t rd_size;
        switch (rs)
        {
        case read_state::READ_MSG_SIZE:
        {
            rd_size = read(fd, reinterpret_cast<uint8_t *>(&msg_size) + read_total, sizeof(msg_size) - read_total);
            if (rd_size == -1 || rd_size == 0)
            {
                LOG_LOCATION("read failed");
                throw std::runtime_error(client_id + ": read failed");
            }
            read_total += rd_size;
            if (read_total != sizeof(msg_size))
                continue;
            break;
        }
        case read_state::READ_MSG_PAYLOAD:
        {
            buf.resize(msg_size);
            rd_size = read(fd, buf.data() + read_total, msg_size - read_total);
            if (rd_size == -1)
            {
                LOG_LOCATION("read failed");
                throw std::runtime_error(client_id + ": read failed");
            }
            read_total += rd_size;
            if (read_total != msg_size)
                continue;
            break;
        }
        }
        /* increment read status */
        rs = static_cast<read_state>(static_cast<int>(rs) + 1);
        read_total = 0;
    }
    close(_epoll_fd);
    return buf;
}

std::string IsolatedRuntime::AtomicReadMsg(int fd)
{
    enum read_state
    {
        READ_MSG_SIZE,
        READ_MSG_PAYLOAD,
        READ_FINISH
    };
    int _epoll_fd = epoll_create1(0);
    if (_epoll_fd == -1)
    {
        LOG_LOCATION("epoll_create1 failed");
        throw std::runtime_error(client_id + ": epoll_create1");
    }
    struct epoll_event ev = {0};
    ev.data.fd = fd;
    ev.events = EPOLLIN;
    if (epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, fd, &ev))
    {
        LOG_LOCATION("epoll_ctl failed");
        close(_epoll_fd);
        throw std::runtime_error(client_id + ": epoll_ctl");
    }
    uint64_t msg_size = 0;
    read_state rs = read_state::READ_MSG_SIZE;
    ssize_t read_total = 0;
    std::string buf;
    std::lock_guard<std::mutex> lock(atomic_fd_mtx);
    while (rs != read_state::READ_FINISH)
    {
        switch (epoll_wait(_epoll_fd, &ev, 1, timeout_ms))
        {
        case -1:
            LOG_LOCATION("epoll_wait failed");
            close(_epoll_fd);
            throw std::runtime_error(client_id + ": epoll_wait");
        case 0:
            LOG_LOCATION("no data");
            close(_epoll_fd);
            throw std::runtime_error(client_id + ": no data");
        default:
            if (ev.events & EPOLLIN)
                break;
            if (ev.events & (EPOLLERR | EPOLLHUP))
            {
                LOG_LOCATION("invalid fd");
                close(_epoll_fd);
                throw std::runtime_error(client_id + ": invalid fd");
            }
        }
        ssize_t rd_size;
        switch (rs)
        {
        case read_state::READ_MSG_SIZE:
        {
            rd_size = read(fd, reinterpret_cast<uint8_t *>(&msg_size) + read_total, sizeof(msg_size) - read_total);
            if (rd_size == -1 || rd_size == 0)
            {
                LOG_LOCATION("read failed");
                throw std::runtime_error(client_id + ": read failed");
            }
            read_total += rd_size;
            if (read_total != sizeof(msg_size))
                continue;
            break;
        }
        case read_state::READ_MSG_PAYLOAD:
        {
            buf.resize(msg_size);
            rd_size = read(fd, buf.data() + read_total, msg_size - read_total);
            if (rd_size == -1)
            {
                LOG_LOCATION("read failed");
                throw std::runtime_error(client_id + ": read failed");
            }
            read_total += rd_size;
            if (read_total != msg_size)
                continue;
            break;
        }
        }
        /* increment read status */
        rs = static_cast<read_state>(static_cast<int>(rs) + 1);
        read_total = 0;
    }
    close(_epoll_fd);
    return buf;
}

void IsolatedRuntime::WriteMsg(int fd, const google::protobuf::Message *msg)
{
    enum write_state
    {
        WRITE_MSG_SIZE,
        WRITE_MSG_PAYLOAD,
        WRITE_FINISH
    };
    std::string byte_msg = msg->SerializeAsString();
    uint64_t msg_size = byte_msg.size();

    write_state ws = write_state::WRITE_MSG_SIZE;
    while (ws != write_state::WRITE_FINISH)
    {
        switch (ws)
        {
        case write_state::WRITE_MSG_SIZE:
        {
            ssize_t write_size = write(fd, &msg_size, sizeof(msg_size));
            if (write_size != sizeof(msg_size))
            {
                throw std::runtime_error(client_id + ": write");
            }
            break;
        }
        case write_state::WRITE_MSG_PAYLOAD:
        {
            ssize_t write_size = write(fd, byte_msg.data(), msg_size);
            if (write_size != msg_size)
            {
                LOG_LOCATION("write failed");
                throw std::runtime_error(client_id + ": write");
            }
            break;
        }
        }
        /* increment read status */
        ws = static_cast<write_state>(static_cast<int>(ws) + 1);
    }
}

void IsolatedRuntime::AtomicWriteMsg(int fd, const google::protobuf::Message *msg)
{
    enum write_state
    {
        WRITE_MSG_SIZE,
        WRITE_MSG_PAYLOAD,
        WRITE_FINISH
    };
    std::string byte_msg = msg->SerializeAsString();
    uint64_t msg_size = byte_msg.size();
    write_state ws = write_state::WRITE_MSG_SIZE;
    std::lock_guard<std::mutex> lock(atomic_fd_mtx);
    while (ws != write_state::WRITE_FINISH)
    {
        switch (ws)
        {
        case write_state::WRITE_MSG_SIZE:
        {
            ssize_t write_size = write(fd, &msg_size, sizeof(msg_size));
            if (write_size != sizeof(msg_size))
            {
                LOG_LOCATION("write failed");
                throw std::runtime_error(client_id + ": write");
            }
            break;
        }
        case write_state::WRITE_MSG_PAYLOAD:
        {
            ssize_t write_size = write(fd, byte_msg.data(), msg_size);
            if (write_size != msg_size)
            {
                LOG_LOCATION("write failed");
                throw std::runtime_error(client_id + ": write");
            }
            break;
        }
        }
        /* increment write status */
        ws = static_cast<write_state>(static_cast<int>(ws) + 1);
    }
}

void IsolatedRuntime::handle_request(int thread_key)
{
    while (true)
    {
        uint64_t msg_size = 0;
        IsolatedRuntime::REQUEST_CODE rc;
        std::vector<char> msg_buf(0);
        std::string byte_msg;
        std::unique_lock<std::mutex> lock(runtime_mtx);
        struct epoll_event ev = {0};

        cv.wait(lock, [this]
                { return !job_queue.empty() || state == RUNTIME_STATE::TERMINATING; });

        if (state == RUNTIME_STATE::TERMINATING)
        {
            cv.notify_all();
            break;
        }

        active_workers++;
        int fd = job_queue.front();
        job_queue.pop();
        lock.unlock();
        int _epoll_fd = epoll_create1(0);
        if (_epoll_fd == -1)
        {
            LOG_LOCATION("epoll_create1 failed");
            {
                std::lock_guard<std::mutex> lock(runtime_mtx);
                state = RUNTIME_STATE::TERMINATING;
            }
            goto clean;
        }
        ev.data.fd = fd;
        ev.events = EPOLLIN;
        if (epoll_ctl(_epoll_fd, EPOLL_CTL_ADD, fd, &ev))
        {
            LOG_LOCATION("epoll_ctl failed");
            {
                std::lock_guard<std::mutex> lock(runtime_mtx);
                state = RUNTIME_STATE::TERMINATING;
            }
            goto clean;
        }
        /* check file descripter */
        std::memset(&ev, 0, sizeof(ev));
        switch (epoll_wait(_epoll_fd, &ev, 1, timeout_ms))
        {
        case -1:
            LOG_LOCATION("epoll_wait failed");
            {
                std::lock_guard<std::mutex> lock(runtime_mtx);
                state = RUNTIME_STATE::TERMINATING;
            }
            goto clean;
        case 0:
            LOG_LOCATION("no message, timeout");
            {
                std::lock_guard<std::mutex> lock(runtime_mtx);
                state = RUNTIME_STATE::TERMINATING;
            }
            goto clean;
        default:
            if (ev.events & (EPOLLHUP | EPOLLERR))
            {
                LOG_LOCATION("invalid fd");
                {
                    std::lock_guard<std::mutex> lock(runtime_mtx);
                    state = RUNTIME_STATE::TERMINATING;
                    LOG_LOCATION("state updated");
                }
                goto clean;
            }
            /* read request code */
            ssize_t real_read_size = read(fd, &rc, sizeof(rc));
            if (real_read_size != sizeof(rc))
            {
                LOG_LOCATION("read failed");
                {
                    std::lock_guard<std::mutex> lock(runtime_mtx);
                    state = RUNTIME_STATE::TERMINATING;
                    LOG_LOCATION("state updated");
                }
                goto clean;
            }
        }
        try
        {
            if (to_underlying(rc) >= handler_table.size())
            {
                LOG_LOCATION("Invalid Request Code");
                throw std::runtime_error("Invalid Request Code");
            }
            (this->*handler_table[to_underlying(rc)])(fd);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            {
                std::lock_guard<std::mutex> lock(runtime_mtx);
                state = RUNTIME_STATE::TERMINATING;
                LOG_LOCATION("state updated");
            }
        }
    clean:
        close(_epoll_fd);
        close(fd);
    decrement_active_workers:
        lock.lock();
        active_workers--;
        if (state == RUNTIME_STATE::TERMINATING)
        {
            cv.notify_all();
            break;
        }
    }

    /* scale-in worker */
    std::lock_guard<std::mutex> lock(runtime_mtx);
    if (state == RUNTIME_STATE::RUNNING)
        workers.erase(thread_key);
}

void IsolatedRuntime::HandleLoadLib(int fd)
{
    void *handler;
    x64::LoadLibMsg msg;
    std::string byte_msg;

    try
    {
        byte_msg = AtomicReadMsg(fd);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }

    if (!msg.ParseFromString(byte_msg))
    {
        LOG_LOCATION("ParseFromString failed");
        msg.mutable_header()->set_msg_type(LOADLIB);
        msg.mutable_header()->set_status(StatusCode::DATA_LOSS);
        msg.mutable_header()->set_client_id(client_id);
        msg.mutable_header()->set_payload_size(0);
        msg.mutable_library_name()->clear();
        msg.mutable_addr2sym()->Clear();
        try
        {
            AtomicWriteMsg(fd, &msg);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
        return;
    }
    handler = dlopen(msg.library_name().c_str(), RTLD_LAZY);
    if (!handler)
    {
        LOG_LOCATION("dlopen failed: " << msg.library_name());
        msg.mutable_header()->set_status(StatusCode::NOT_FOUND);
        try
        {
            AtomicWriteMsg(fd, &msg);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
        return;
    }
    sym_handler.push_back(handler);
    for (int addr2sym = 0; addr2sym < msg.addr2sym().size(); addr2sym++)
        this->addr2sym[msg.addr2sym().at(addr2sym).address()] = msg.addr2sym().at(addr2sym).name();
    msg.mutable_header()->set_status(StatusCode::OK);
    msg.mutable_header()->set_payload_size(0);
    msg.mutable_library_name()->clear();
    msg.mutable_addr2sym()->Clear();
    try
    {
        AtomicWriteMsg(fd, &msg);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }
    return;
}

void IsolatedRuntime::HandleInvokeFunc(int fd)
{
    const int cpustate_size = sizeof(gregset_t) + sizeof(_libc_fpstate);
    uint64_t write_back_page_size = 0;
    void *ip = nullptr;
    void *return_addr = &&get_ctx;
    x64::CPUState cpu;
    x64::X64FPRegs fpregs;
    std::string byte_msg;
    InvokeFuncMsg msg, resp;
    unsigned long long fs_base;
    const uint64_t *gregs;
    std::list<shared_ptr<x64::Page>> release_page_list;

    try
    {
        byte_msg = AtomicReadMsg(fd);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }
    if (!msg.ParseFromString(byte_msg))
    {
        LOG_LOCATION("ParseFromString failed");
        resp.mutable_header()->set_msg_type(INVOKEFUNC);
        resp.mutable_header()->set_status(StatusCode::DATA_LOSS);
        resp.mutable_header()->set_client_id(client_id);
        resp.mutable_header()->set_payload_size(0);
        try
        {
            AtomicWriteMsg(fd, &resp);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
        return;
    }

    resp.mutable_header()->CopyFrom(msg.header());

    /* check instruction pointer */
    if (addr2sym.find(msg.ctx().cpu().gregs()[REG_RIP]) == addr2sym.end())
    {
        ip = reinterpret_cast<void *>(
            msg.ctx().cpu().gregs()[REG_RIP]);
        /* check instruction pointer address */
        void *try_mmap = mmap(
            reinterpret_cast<void *>(
                reinterpret_cast<uint64_t>(ip) & RUNTIME_H_PAGE_MASK),
            page_size,
            PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS,
            -1, 0);
        munmap(try_mmap, page_size);
        if (reinterpret_cast<void *>(
                reinterpret_cast<uint64_t>(ip) & RUNTIME_H_PAGE_MASK) == try_mmap)
        {
            LOG_LOCATION("Invalid Instruction Pointer");
            resp.mutable_header()->set_status(StatusCode::NOT_FOUND);
            resp.mutable_header()->set_payload_size(0);
            try
            {
                AtomicWriteMsg(fd, &resp);
            }
            catch (const std::exception &e)
            {
                LOG_LOCATION(e.what());
                throw std::runtime_error(e.what());
            }
            return;
        }
        LOG_LOCATION("call unknown symbol at " << ip);
        goto set_context;
    }

    /* rewrite instruction pointer */
    for (std::list<void *>::iterator handler = sym_handler.begin();
         !ip && handler != sym_handler.end(); handler++)
    {
        ip = dlsym(*handler,
                   addr2sym[msg.ctx().cpu().gregs()[REG_RIP]].c_str());
    }
    if (!ip)
    {
        LOG_LOCATION("dlsym failed");
        resp.mutable_header()->set_status(StatusCode::NOT_FOUND);
        resp.mutable_header()->set_payload_size(0);
        try
        {
            AtomicWriteMsg(fd, &resp);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
        return;
    }

set_context:
    LOG_LOCATION("called " << addr2sym[msg.ctx().cpu().gregs()[REG_RIP]].c_str());
    try
    {
        page_manager.FlushClientChange(msg.page(), msg.ctx().stack_bottom());
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        resp.mutable_header()->set_status(StatusCode::INTERNAL);
        resp.mutable_header()->set_payload_size(0);
        resp.mutable_ctx()->CopyFrom(msg.ctx());
        try
        {
            AtomicWriteMsg(fd, &resp);
        }
        catch (const std::exception &e2)
        {
            LOG_LOCATION(e2.what());
            throw std::runtime_error(e2.what());
        }
        throw std::runtime_error(e.what());
    }

    page_manager.WaitPrevClientFlush(msg.invokefunc_id());

    /* rewrite return address */
    std::memcpy(
        reinterpret_cast<void *>(
            msg.ctx().cpu().gregs().at(REG_RSP)),
        &return_addr, sizeof(void *));

    /* save gregs in current stack frame */
    gregs = msg.ctx().cpu().gregs().data();

    g_fpstate.cwd = msg.ctx().cpu().fpregs().cwd();
    g_fpstate.swd = msg.ctx().cpu().fpregs().swd();
    g_fpstate.ftw = msg.ctx().cpu().fpregs().ftw();
    g_fpstate.fop = msg.ctx().cpu().fpregs().fop();
    g_fpstate.rip = msg.ctx().cpu().fpregs().rip();
    g_fpstate.rdp = msg.ctx().cpu().fpregs().rdp();
    g_fpstate.mxcsr = msg.ctx().cpu().fpregs().mxcsr();
    g_fpstate.mxcr_mask = msg.ctx().cpu().fpregs().mxcr_mask();

    for (int i = 0; i < RUNTIME_H_ARRAY_SIZE(g_fpstate._st); i++)
    {
        std::memcpy(g_fpstate._st[i].significand,
                    msg.ctx().cpu().fpregs().st().at(i).significand().data(),
                    sizeof(g_fpstate._st[i].significand));
        uint16_t exponent = msg.ctx().cpu().fpregs().st().at(i).exponent();
        std::memcpy(&g_fpstate._st[i].exponent, &exponent,
                    sizeof(g_fpstate._st[i].exponent));
        std::memcpy(g_fpstate._st[i].__glibc_reserved1,
                    msg.ctx().cpu().fpregs().st().at(i).reserved().data(),
                    sizeof(g_fpstate._st[i].__glibc_reserved1));
    }
    for (int i = 0; i < RUNTIME_H_ARRAY_SIZE(g_fpstate._xmm); i++)
        std::memcpy(g_fpstate._xmm[i].element,
                    msg.ctx().cpu().fpregs().xmm().at(i).element().data(),
                    sizeof(g_fpstate._xmm[i].element));

    /* set tls variable */
    request_info.request_msg = &msg;
    request_info.fd = fd;
    request_info.rt_object = this;
    request_info.client_id = &client_id;

    /* get current fsbase */
    fs_base = _readfsbase_u64();
    /* save current fsbase */
    _writegsbase_u64(fs_base);

    /* restore fpstate */
    asm volatile(
        "fxrstor (%%rax)"
        :
        : "a"(&g_fpstate)
        :);

    /* restore flag register */
    asm volatile(
        "push %%rax\n\t"
        "popfq\n\t"
        :
        : "a"(gregs[REG_EFL]));

    /* save current stack frame info */
    /* save current rsp */
    asm volatile(
        "mov %%rsp, (%%rax)"
        :
        : "a"(&g_greg[REG_RSP]));
    /* save current rbp */
    asm volatile(
        "mov %%rbp, (%%rax)"
        :
        : "a"(&g_greg[REG_RBP]));

#define RESTORE_GREG(r, R)    \
    asm volatile(             \
        "mov %0," r           \
        :                     \
        : "m"(gregs[REG_##R]) \
        :)

    RESTORE_GREG("%%r8", R8);
    RESTORE_GREG("%%r9", R9);
    RESTORE_GREG("%%r10", R10);
    RESTORE_GREG("%%r11", R11);
    RESTORE_GREG("%%r12", R12);
    RESTORE_GREG("%%r13", R13);
    RESTORE_GREG("%%r14", R14);
    RESTORE_GREG("%%r15", R15);
    RESTORE_GREG("%%rdi", RDI);
    RESTORE_GREG("%%rsi", RSI);
    RESTORE_GREG("%%rbx", RBX);
    RESTORE_GREG("%%rdx", RDX);
    RESTORE_GREG("%%rcx", RCX);
    RESTORE_GREG("%%rsp", RSP);
    asm volatile(
        "push %%rax"
        :
        : "a"(ip));
    asm volatile(
        "push %%rax"
        :
        : "a"(gregs[REG_RBP]));
    RESTORE_GREG("%%rax", RAX);

    asm volatile(
        "pop %rbp\n\t"
        "ret");
#undef RESTORE_GREG

get_ctx:

    /* save non callee saved regs */
    asm volatile(
        "push %r8\n\t"
        "push %r9\n\t"
        "push %r10\n\t"
        "push %r11\n\t"
        "push %r12\n\t"
        "push %r13\n\t"
        "push %r14\n\t"
        "push %r15\n\t"
        "push %rdi\n\t"
        "push %rsi\n\t"
        "push %rdx\n\t"
        "push %rax\n\t"
        "push %rcx\n\t"
        "pushfq\n\t");

    /* restore stack frame */
    asm volatile(
        "mov %0, %%rsp"
        :
        : "m"(g_greg[REG_RSP]));
    asm volatile(
        "mov %0, %%rbp"
        :
        : "m"(g_greg[REG_RBP]));
    /* now, we can access local variables */

    /* save fpstate */
    asm volatile("fxsave %0" ::"m"(g_fpstate));

    fpregs.set_cwd(g_fpstate.cwd);
    fpregs.set_swd(g_fpstate.swd);
    fpregs.set_ftw(g_fpstate.ftw);
    fpregs.set_fop(g_fpstate.fop);
    fpregs.set_rip(g_fpstate.rip);
    fpregs.set_rdp(g_fpstate.rdp);
    fpregs.set_mxcsr(g_fpstate.mxcsr);
    fpregs.set_mxcr_mask(g_fpstate.mxcr_mask);

    /* add x87 values to the response object */
    for (int st_index = 0; st_index < msg.ctx().cpu().fpregs().st().size(); st_index++)
    {
        X64FPXReg st;
        for (int i = 0; i < RUNTIME_H_ARRAY_SIZE(g_fpstate._st[st_index].significand); i++)
            st.add_significand(g_fpstate._st[st_index].significand[i]);

        st.set_exponent(g_fpstate._st[st_index].exponent);

        for (int i = 0; i < RUNTIME_H_ARRAY_SIZE(g_fpstate._st[st_index].__glibc_reserved1); i++)
            st.add_reserved(g_fpstate._st[st_index].__glibc_reserved1[i]);

        fpregs.add_st()->CopyFrom(st);
    }

    /* add xmm values to the response object */
    for (int i = 0; i < RUNTIME_H_ARRAY_SIZE(g_fpstate._xmm); i++)
        fpregs.add_xmm()
            ->mutable_element()
            ->Add(
                g_fpstate._xmm[i].element,
                g_fpstate._xmm[i].element +
                    RUNTIME_H_ARRAY_SIZE(g_fpstate._xmm[i].element));

    for (int reserved = 0;
         reserved < msg.ctx().cpu().fpregs().reserved().size();
         reserved++)
        fpregs.add_reserved(
            msg.ctx().cpu().fpregs().reserved()[reserved]);

    cpu.mutable_fpregs()->CopyFrom(fpregs);

    // set greg value
    uint64_t *src_greg_addr = reinterpret_cast<uint64_t *>(
        msg.ctx().cpu().gregs().at(REG_RSP));
    cpu.add_gregs(*src_greg_addr--); /* R8 */
    cpu.add_gregs(*src_greg_addr--); /* R9 */
    cpu.add_gregs(*src_greg_addr--); /* R10 */
    cpu.add_gregs(*src_greg_addr--); /* R11 */
    cpu.add_gregs(*src_greg_addr--); /* R12 */
    cpu.add_gregs(*src_greg_addr--); /* R13 */
    cpu.add_gregs(*src_greg_addr--); /* R14 */
    cpu.add_gregs(*src_greg_addr--); /* R15 */
    cpu.add_gregs(*src_greg_addr--); /* RDI */
    cpu.add_gregs(*src_greg_addr--); /* RSI */
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_RBP));
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_RBX));
    cpu.add_gregs(*src_greg_addr--); /* RDX */
    cpu.add_gregs(*src_greg_addr--); /* RAX */
    cpu.add_gregs(*src_greg_addr--); /* RCX */
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_RSP));
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_RIP));
    cpu.add_gregs(*src_greg_addr--); /* EFL */
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_CSGSFS));
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_ERR));
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_TRAPNO));
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_OLDMASK));
    cpu.add_gregs(msg.ctx().cpu().gregs().at(REG_CR2));
    resp.mutable_ctx()->mutable_cpu()->CopyFrom(cpu);

    {
        std::lock_guard<std::mutex> lock(resp_id_mtx);
        std::vector<x64::Page> diff_list = page_manager.FlushLocalChange();
        for (x64::Page diff : diff_list)
            resp.add_page()->CopyFrom(diff);

        for (shared_ptr<x64::Page> release_page : release_page_list)
            resp.add_page()->CopyFrom(*release_page);
        resp.set_resp_id(resp_id++);
    }

    resp.mutable_header()->set_status(StatusCode::OK);
    try
    {
        AtomicWriteMsg(fd, &resp);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }
    return;
}

void IsolatedRuntime::HandlePullPage(int fd)
{
    uint64_t payload_size = 0;
    std::string byte_msg;
    x64::PullPageMsg msg, resp;

    try
    {
        byte_msg = AtomicReadMsg(fd);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }
    if (!msg.ParseFromString(byte_msg))
    {
        msg.mutable_header()->set_msg_type(PULLPAGE);
        msg.mutable_header()->set_status(StatusCode::DATA_LOSS);
        msg.mutable_header()->set_client_id(client_id);
        msg.mutable_header()->set_payload_size(0);
        try
        {
            AtomicWriteMsg(fd, &msg);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
    }
    if (msg.page().size() == 0)
    {
        msg.mutable_header()->set_msg_type(PULLPAGE);
        msg.mutable_header()->set_status(StatusCode::NOT_FOUND);
        msg.mutable_header()->set_client_id(client_id);
        msg.mutable_header()->set_payload_size(0);
        try
        {
            AtomicWriteMsg(fd, &msg);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            throw std::runtime_error(e.what());
        }
    }
    resp.CopyFrom(msg);
    resp.clear_page();
    for (x64::Page p : msg.page())
    {
        payload_size += sizeof(p.address()) + sizeof(p.content_size());
        x64::Page page;
        uint64_t page_start = p.address() & RUNTIME_H_PAGE_MASK;
        page.set_address(page_start);
        page.set_runtime_revision(0);
        page.set_client_revision(0);
        page.set_content_size(page_size);

        /* check if requested page is mapped or not */
        void *try_mmap = mmap(reinterpret_cast<void *>(page_start), page_size,
                              PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        munmap(try_mmap, page_size);
        if (try_mmap == reinterpret_cast<void *>(page_start))
        {
            LOG_LOCATION("invalid page requested " << reinterpret_cast<void *>(page_start));
            page.set_content_size(0); /* request page isn't mapped */
            resp.add_page()->CopyFrom(page);
            resp.mutable_header()->set_status(StatusCode::NOT_FOUND);
            resp.mutable_header()->set_payload_size(payload_size);
            goto reply;
        }
        payload_size += page_size;
        page.set_allocated_content(new std::string(
            reinterpret_cast<char *>(page_start), page_size));
        resp.add_page()->CopyFrom(page);
        page_manager.AddPushPage(page);
    }
    resp.mutable_header()->set_payload_size(payload_size);
    resp.mutable_header()->set_status(StatusCode::OK);
reply:
    try
    {
        AtomicWriteMsg(fd, &resp);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }
}

int IsolatedRuntime::ConnectRuntimeSocket()
{
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1)
    {
        LOG_LOCATION("socket failed");
        return sock;
    }
    struct sockaddr_un endpoint = {0};
    endpoint.sun_family = AF_UNIX;
    std::memcpy(endpoint.sun_path, sun_path.c_str(), std::strlen(sun_path.c_str()));
    if (connect(sock, reinterpret_cast<struct sockaddr *>(&endpoint),
                sizeof(endpoint)) == -1)
    {
        LOG_LOCATION("connect failed");
        close(sock);
        return -1;
    }
    return sock;
}

std::string IsolatedRuntime::DispatchHandler(int fd, const google::protobuf::Message *request, REQUEST_CODE request_code)
{
    try
    {
        if (write(fd, &request_code, sizeof(request_code)) == -1)
        {
            LOG_LOCATION("write failed");
            throw std::runtime_error(client_id + ": write");
        }
        WriteMsg(fd, request);
        return ReadMsg(fd);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        throw std::runtime_error(e.what());
    }
    return "";
}

Status X64SigRPCService::LoadLib(ServerContext *context, const LoadLibMsg *request, LoadLibMsg *response)
{

    std::string client_id = request->header().client_id();
    response->mutable_header()->set_msg_type(request->header().msg_type());
    response->mutable_header()->set_client_id(client_id);
    std::shared_ptr<IsolatedRuntime> stub;
    std::unique_lock<std::shared_mutex> writer_lock(runtimes_map_mtx);

    if (runtimes.find(client_id) == runtimes.end()) /* no runtime */
    {
        /* add new runtime */
        try
        {
            runtimes[client_id].second = std::make_shared<IsolatedRuntime>(client_id, sock_dir, 30 * 1000);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            response->mutable_header()->set_status(StatusCode::INTERNAL);
            response->mutable_header()->set_payload_size(0);
            return Status(StatusCode::INTERNAL, e.what());
        }

        stub = runtimes.at(client_id).second;

        int wait_listen[2];
        if (pipe(wait_listen) == -1)
        {
            LOG_LOCATION("pipe failed");
            runtimes.erase(client_id);
            response->mutable_header()->set_status(StatusCode::INTERNAL);
            response->mutable_header()->set_payload_size(0);
            return Status(StatusCode::INTERNAL, client_id + ": pipe failed");
        }

        RAIIFD sync_epoll_fd(epoll_create1(0));
        if (sync_epoll_fd.fd == -1)
        {
            LOG_LOCATION("epoll_create1 failed errno = " << errno);
            close(wait_listen[0]);
            close(wait_listen[1]);
            runtimes.erase(client_id);
            response->mutable_header()->set_status(StatusCode::INTERNAL);
            response->mutable_header()->set_payload_size(0);
            return Status(StatusCode::INTERNAL, client_id + ": epoll_create1 failed");
        }
        struct epoll_event ev = {0};
        ev.data.fd = wait_listen[0];
        ev.events = EPOLLIN;
        if (epoll_ctl(sync_epoll_fd.fd, EPOLL_CTL_ADD, wait_listen[0], &ev))
        {
            LOG_LOCATION("epoll_ctl failed");
            close(wait_listen[0]);
            close(wait_listen[1]);
            runtimes.erase(client_id);
            response->mutable_header()->set_status(StatusCode::INTERNAL);
            response->mutable_header()->set_payload_size(0);
            return Status(StatusCode::INTERNAL, client_id + ": epoll_ctl failed");
        }

        int fork_state = fork();
        if (fork_state == -1)
        {
            LOG_LOCATION("fork failed");
            close(wait_listen[0]);
            close(wait_listen[1]);
            runtimes.erase(client_id);
            response->mutable_header()->set_status(StatusCode::INTERNAL);
            response->mutable_header()->set_payload_size(0);
            return Status(StatusCode::INTERNAL, client_id + ": fork failed");
        }

        if (fork_state == 0) /* child process */
        {
            struct rlimit limit;
            if (getrlimit(RLIMIT_NOFILE, &limit))
            {
                LOG_LOCATION("getrlimit failed");
                std::exit(EXIT_FAILURE);
            }
            /* avoid to receive event */
            close_range(3, wait_listen[1] - 1, 0);
            close_range(wait_listen[1] + 1, limit.rlim_max, 0);

            /* start to listen socket */
            try
            {
                /* block until runtime is shutdown */
                stub->Serve(min_runtime_workers, wait_listen[1]);
            }
            catch (const std::exception &e)
            {
                LOG_LOCATION(e.what());
            }
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            close(wait_listen[1]);
            stub = runtimes.at(client_id).second;
            stub->SetRuntime(fork_state);

            if (!stub->WaitRuntime(sync_epoll_fd.fd))
            {
                close(wait_listen[0]);
                runtimes.erase(client_id);
                LOG_LOCATION("WaitRuntime failed");
                response->mutable_header()->set_status(StatusCode::INTERNAL);
                response->mutable_header()->set_payload_size(0);
                return Status(StatusCode::INTERNAL, client_id + ": WaitRuntime failed");
            }
            close(wait_listen[0]);
            int sock_fd = stub->HookTerminate(epoll_fd);
            if (sock_fd == -1)
            {
                runtimes.erase(client_id);
                LOG_LOCATION("HookTerminate failed");
                response->mutable_header()->set_status(StatusCode::INTERNAL);
                response->mutable_header()->set_payload_size(0);
                return Status(StatusCode::INTERNAL, client_id + ": HookTerminate failed");
            }
            runtimes.at(client_id).first = sock_fd;
            if (fd2runtime.find(sock_fd) != fd2runtime.end())
            {
                LOG_LOCATION("conflict fd2runtime");
                std::string old_client_id = fd2runtime.at(sock_fd);
                fd2runtime.erase(sock_fd);
                runtimes.erase(old_client_id);
                LOG_LOCATION("resolved fd2runtime");
            }
            fd2runtime[sock_fd] = client_id;
        }
    }
    StatusCode status;
    std::string description;
    stub = runtimes.at(client_id).second;
    writer_lock.unlock();
    RAIIFD client_sock(-1);
    std::string byte_response;
    client_sock.fd = stub->ConnectRuntimeSocket();
    if (client_sock.fd == -1)
    {
        LOG_LOCATION("ConnectRuntimeSocket failed");
        writer_lock.lock();
        fd2runtime.erase(runtimes.at(client_id).first);
        runtimes.erase(client_id);
        response->mutable_header()->set_status(StatusCode::INTERNAL);
        response->mutable_header()->set_payload_size(0);
        return Status(StatusCode::INTERNAL, client_id + ": ConnectRuntimeSocket failed");
    }
    try
    {
        byte_response = stub->DispatchHandler(client_sock.fd, request, IsolatedRuntime::REQUEST_CODE::LOADLIB);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        response->mutable_header()->set_status(StatusCode::INTERNAL);
        response->mutable_header()->set_payload_size(0);
        writer_lock.lock();
        fd2runtime.erase(runtimes.at(client_id).first);
        runtimes.erase(client_id);
        status = static_cast<StatusCode>(response->header().status());
        if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
            description = IsolatedRuntime::StatusDescription.at(status);
        return Status(status, description);
    }
    if (!response->ParseFromString(byte_response))
    {
        LOG_LOCATION("ParseFromString failed");
        response->mutable_header()->set_status(StatusCode::DATA_LOSS);
        response->mutable_header()->set_payload_size(0);
    }
    status = static_cast<StatusCode>(response->header().status());
    if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
        description = IsolatedRuntime::StatusDescription.at(status);
    switch (status)
    {
    case StatusCode::OK:
        break;
    default:
        writer_lock.lock();
        fd2runtime.erase(runtimes.at(client_id).first);
        runtimes.erase(client_id);
    }
    return Status(status, description);
}

Status X64SigRPCService::InvokeFunc(
    ServerContext *context,
    grpc::ServerReaderWriter<InvokeFuncMsg, InvokeFuncMsg> *stream)
{
    StatusCode status;
    std::string description;
    InvokeFuncMsg msg;
    std::string client_id;
    RAIIFD client_sock(-1);
    std::shared_ptr<IsolatedRuntime> stub = nullptr;

    while (stream->Read(&msg))
    {
        if (!stub)
        {
            std::shared_lock<std::shared_mutex> readers_lock(runtimes_map_mtx);
            client_id = msg.header().client_id();
            if (runtimes.find(client_id) == runtimes.end())
            {
                LOG_LOCATION("runtime not found");
                msg.mutable_header()->set_status(StatusCode::NOT_FOUND);
                msg.mutable_header()->set_payload_size(0);
                status = static_cast<StatusCode>(msg.header().status());
                if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
                    description = IsolatedRuntime::StatusDescription.at(status);
                if (stream->Write(msg))
                    LOG_LOCATION("stream closed");
                return Status(status, description);
            }
            stub = runtimes.at(client_id).second;
            readers_lock.unlock();
            client_sock.fd = stub->ConnectRuntimeSocket();
            if (client_sock.fd == -1)
            {
                LOG_LOCATION("ConnectRuntimeSocket failed");
                msg.mutable_header()->set_status(StatusCode::INTERNAL);
                msg.mutable_header()->set_payload_size(0);
                status = static_cast<StatusCode>(msg.header().status());
                if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
                    description = IsolatedRuntime::StatusDescription.at(status);
                if (stream->Write(msg))
                    LOG_LOCATION("stream closed");
                std::lock_guard<shared_mutex> writer_lock(runtimes_map_mtx);
                fd2runtime.erase(runtimes.at(client_id).first);
                runtimes.erase(client_id);
                return Status(status, description);
            }
        }
        std::string byte_response;
        try
        {
            byte_response = stub->DispatchHandler(
                client_sock.fd, &msg, IsolatedRuntime::REQUEST_CODE::INVOKEFUNC);
        }
        catch (const std::exception &e)
        {
            LOG_LOCATION(e.what());
            status = static_cast<StatusCode>(msg.header().status());
            if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
                description = IsolatedRuntime::StatusDescription.at(status);
            std::lock_guard<std::shared_mutex> lock(runtimes_map_mtx);
            fd2runtime.erase(runtimes.at(client_id).first);
            runtimes.erase(client_id);
            return Status(status, description);
        }
        if (!msg.ParseFromString(byte_response))
        {
            LOG_LOCATION("ParseFromString failed");
            msg.mutable_header()->set_msg_type(INVOKEFUNC);
            msg.mutable_header()->set_status(StatusCode::DATA_LOSS);
            msg.mutable_header()->set_client_id(client_id);
            msg.mutable_header()->set_payload_size(0);
        }
        status = static_cast<StatusCode>(msg.header().status());
        if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
            description = IsolatedRuntime::StatusDescription.at(status);
        switch (status)
        {
        case StatusCode::OK: /* invokefunc finished correctly */
            if (stream->Write(msg))
            {
                return Status(status, description);
            }
            else
            {
                LOG_LOCATION("stream closed");
                std::lock_guard<shared_mutex> writer_lock(runtimes_map_mtx);
                fd2runtime.erase(runtimes.at(client_id).first);
                runtimes.erase(client_id);
                return Status(status, description);
            }
        case StatusCode::NOT_FOUND: /* SEGV handler called ? */
            if (stream->Write(msg))
                break; /* do not return, read next message */
            else
            {
                LOG_LOCATION("stream closed");
                std::lock_guard<shared_mutex> writer_lock(runtimes_map_mtx);
                fd2runtime.erase(runtimes.at(client_id).first);
                runtimes.erase(client_id);
                return Status(status, description);
            }
        default:
            if (!stream->Write(msg))
                LOG_LOCATION("stream closed");
            std::lock_guard<shared_mutex> writer_lock(runtimes_map_mtx);
            fd2runtime.erase(runtimes.at(client_id).first);
            runtimes.erase(client_id);
            return Status(status, description);
        }
    }
    if (context->IsCancelled())
    {
        msg.mutable_header()->set_msg_type(INVOKEFUNC);
        msg.mutable_header()->set_payload_size(0);
        msg.mutable_header()->set_status(StatusCode::DATA_LOSS);
        if (!stream->Write(msg))
            LOG_LOCATION("stream closed");
        status = static_cast<StatusCode>(msg.header().status());
        if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
            description = IsolatedRuntime::StatusDescription.at(status);
        std::lock_guard<std::shared_mutex> lock(runtimes_map_mtx);
        fd2runtime.erase(runtimes.at(client_id).first);
        runtimes.erase(client_id);
        return Status(status, description);
    }
    status = StatusCode::OK;
    if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
        description = IsolatedRuntime::StatusDescription.at(status);
    return Status(status, description);
}

Status
X64SigRPCService::PullPage(
    ServerContext *context,
    const PullPageMsg *request,
    PullPageMsg *response)
{
    std::string client_id = request->header().client_id();
    response->mutable_header()->set_msg_type(request->header().msg_type());
    response->mutable_header()->set_client_id(client_id);
    StatusCode status;
    std::string description;
    std::shared_lock<std::shared_mutex> readers_lock(runtimes_map_mtx);
    if (runtimes.find(client_id) == runtimes.end())
    {
        LOG_LOCATION("runtime not found");
        response->mutable_header()->set_status(StatusCode::NOT_FOUND);
        response->mutable_header()->set_payload_size(0);
        status = static_cast<StatusCode>(response->header().status());
        if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
            description = IsolatedRuntime::StatusDescription.at(status);
        return Status(status, description);
    }
    std::shared_ptr<IsolatedRuntime> stub = runtimes.at(client_id).second;
    readers_lock.unlock();
    RAIIFD client_sock(-1);
    std::string byte_response;
    client_sock.fd = stub->ConnectRuntimeSocket();
    if (client_sock.fd == -1)
    {
        LOG_LOCATION("ConnectRuntimeSocket failed");
        std::lock_guard<std::shared_mutex> writer_lock(runtimes_map_mtx);
        fd2runtime.erase(runtimes.at(client_id).first);
        runtimes.erase(client_id);
        response->mutable_header()->set_status(StatusCode::INTERNAL);
        response->mutable_header()->set_payload_size(0);
        return Status(StatusCode::INTERNAL, client_id + ": ConnectRuntimeSocket failed");
    }
    try
    {
        byte_response = stub->DispatchHandler(client_sock.fd, request, IsolatedRuntime::REQUEST_CODE::PULLPAGE);
    }
    catch (const std::exception &e)
    {
        LOG_LOCATION(e.what());
        std::lock_guard<std::shared_mutex> writer_lock(runtimes_map_mtx);
        fd2runtime.erase(runtimes.at(client_id).first);
        runtimes.erase(client_id);
        status = static_cast<StatusCode>(response->header().status());
        if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
            description = IsolatedRuntime::StatusDescription.at(status);
        return Status(status, description);
    }
    if (!response->ParseFromString(byte_response))
    {
        LOG_LOCATION("ParseFromString failed");
        response->mutable_header()->set_status(StatusCode::INTERNAL);
        response->mutable_header()->set_payload_size(0);
    }
    status = static_cast<StatusCode>(response->header().status());
    if (IsolatedRuntime::StatusDescription.find(status) != IsolatedRuntime::StatusDescription.end())
        description = IsolatedRuntime::StatusDescription.at(status);
    return Status(status, description);
}

void X64SigRPCService::run_gc()
{
    struct epoll_event event[10];
    while (true)
    {
        int nfds = epoll_wait(epoll_fd, event, RUNTIME_H_ARRAY_SIZE(event), -1);
        if (nfds == -1)
        {
            LOG_LOCATION("epoll_wait failed");
            goto clean;
        }
        for (int i = 0; i < nfds; i++)
        {
            if (event[i].events & (EPOLLHUP | EPOLLERR))
            {
                std::lock_guard<std::shared_mutex> lock(runtimes_map_mtx);
                LOG_LOCATION("runtime count=" << runtimes.size());
                std::string client_id = fd2runtime[event[i].data.fd];
                if (epoll_ctl(epoll_fd, EPOLL_CTL_DEL, event[i].data.fd, NULL) == -1)
                {
                    if (fcntl(epoll_fd, F_GETFD) == -1)
                    {
                        LOG_LOCATION("epoll_ctl failed errno=" << errno);
                        goto clean;
                    }
                }
                LOG_LOCATION("erase " << client_id);
                fd2runtime.erase(event[i].data.fd);
                runtimes.erase(client_id);
            }
        }
    }
clean:
    std::lock_guard<std::shared_mutex> lock(runtimes_map_mtx);
    runtimes.clear();
    std::exit(errno);
}

void RunServer(const char *rpc_listen_addr)
{
    std::string server_address(rpc_listen_addr);
    X64SigRPCService service(4, "/tmp/");

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;

    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS, 4);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MIN_POLLERS, 1);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MAX_POLLERS, 8);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::CQ_TIMEOUT_MSEC, 10);

    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cerr << "SigRPC listening on " << server_address << std::endl;

    server->Wait();
}

int main(void)
{
    const char *rpc_listen_addr = std::getenv("RPC_LISTEN_ADDR");
    if (!rpc_listen_addr)
    {
        std::cerr << "RPC_LISTEN_ADDR is empty" << std::endl;
        return 0;
    }
    if (getauxval(AT_HWCAP2) & HWCAP2_FSGSBASE)
        RunServer(rpc_listen_addr);
    else
        std::cerr << "FSGSBASE disabled" << std::endl;
}
