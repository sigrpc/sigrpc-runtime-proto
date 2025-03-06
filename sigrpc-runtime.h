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

#ifndef RUNTIME_H
#define RUNTIME_H

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif //(_DEFAULT_SOURCE)

#define LOG_LOCATION(log) std::cerr << __FILE__ << ":" << __LINE__ << " " << log << std::endl

#include <unistd.h> // getpagesize

#include <string>
#include <cstring>
#include <cerrno>
#include <map>
#include <list>
#include <vector>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <stdexcept>
#include <cstdlib>
#include <condition_variable>
#include <queue>

#include <signal.h>
#include <dlfcn.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>

using namespace std;

#include "message.pb.h"
#include "message.grpc.pb.h"
using namespace x64;
using grpc::Status;
using grpc::StatusCode;

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpc/grpc.h>
#include <grpcpp/security/credentials.h>
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Service;
using grpc::Status;

enum
{
    LOADLIB,
    INVOKEFUNC,
    PULLPAGE,
};

#define RUNTIME_H_ARRAY_SIZE(arr) sizeof(arr) / sizeof(arr[0])
#define RUNTIME_H_PAGE_MASK 0xfffffffffffff000

class RAIIFD
{
public:
    RAIIFD(int fd) : fd(fd) {}
    ~RAIIFD()
    {
        if (fd != -1)
            close(fd);
    }
    int fd;
};

class PageManager
{
public:
    void FlushClientChange(const google::protobuf::RepeatedPtrField<x64::Page> src, uint64_t stack_bottom);
    std::vector<x64::Page> FlushLocalChange();
    std::string client_id;
    void AddPushPage(x64::Page page);
    bool FindPage(uint64_t page_start);
    void WaitPrevClientFlush(uint64_t invokefunc_id);

private:
    struct successor_revision_s
    {
        std::condition_variable cv;
        bool notified;
    };
    typedef enum
    {
        TYPE_PUSHPAGE,
        TYPE_PULLPAGE,
        TYPE_WAITMMAP,
    } page_type_t;
    struct page_state_s
    {
        std::shared_ptr<x64::Page> page;
        page_type_t page_type;
        std::map<uint64_t, std::shared_ptr<struct successor_revision_s>> successor_list;
    };
    struct mem_region_s
    {
        uint64_t start;
        uint64_t end;
        uint64_t wr_start;
        std::shared_ptr<std::set<struct mem_region_s>> sync_region;
        bool operator<(const struct mem_region_s &other) const
        {
            return start < other.start;
        }
    };
    uint64_t runnable_invokefunc_id = 0;
    void update_client_revision(uint64_t page_start, uint64_t next_client_revision);
    void flush_client_page_change(x64::Page src, uint64_t stack_bottom, std::set<struct mem_region_s> &skip_page_list, std::unique_lock<std::mutex> &lock);
    void flush_client_stack(x64::Page src, uint64_t stack_bottom, std::set<struct mem_region_s> &skip_page_list, std::unique_lock<std::mutex> &lock);
    void flush_client_heap(x64::Page src, std::unique_lock<std::mutex> &lock);
    std::vector<x64::Page> flush_local_page_change(std::shared_ptr<x64::Page> p);
    std::vector<Page> get_page_diff(const void *new_page, void *original_page, size_t size);
    bool in_mem_region(struct mem_region_s mem_region, uint64_t addr);
    bool in_stack_region(x64::Page page);
    void wait_prev_revision(x64::Page page, std::unique_lock<std::mutex> &lock);

    std::map<uint64_t, struct page_state_s> page_state;
    std::set<struct mem_region_s> stack_page_tbl;
    std::set<struct mem_region_s> heap_page_tbl;
    std::mutex map_mtx;
    std::mutex invokefunc_mtx;
    std::unordered_map<uint64_t, std::shared_ptr<std::condition_variable>> invokefunc_mtx_list;
};

class IsolatedRuntime
{
public:
    enum class ROLE
    {
        RUNTIME,
        STUB
    };
    enum class REQUEST_CODE
    {
        LOADLIB,
        INVOKEFUNC,
        PULLPAGE,
        HEALTHCHECK,
        RESERVED
    };
    /* this constructor can throw exception */
    IsolatedRuntime(std::string client_id, std::string sock_dir, int timeout_ms)
    {
        this->fork_state = fork_state;
        this->role = ROLE::STUB;

        if (sock_dir.back() != '/')
            sock_dir.push_back('/');
        sun_path = sock_dir + client_id;
        /*
         * if timeout ms has elapsed since the last request,
         * this runtime is shut down.
         */
        this->timeout_ms = timeout_ms;
        active_workers = 0;
        this->client_id = client_id;
        page_manager.client_id = client_id;
        handler_table[static_cast<int>(REQUEST_CODE::LOADLIB)] = &IsolatedRuntime::HandleLoadLib;
        handler_table[static_cast<int>(REQUEST_CODE::INVOKEFUNC)] = &IsolatedRuntime::HandleInvokeFunc;
        handler_table[static_cast<int>(REQUEST_CODE::PULLPAGE)] = &IsolatedRuntime::HandlePullPage;
        min_workers = 0;
        state = RUNTIME_STATE::CREATED;
    }
    ~IsolatedRuntime()
    {
        switch (role)
        {
        case ROLE::RUNTIME:
        {
            unlink(sun_path.c_str());
            close(epoll_fd);
            close(listen_request_fd);
            state = RUNTIME_STATE::TERMINATING;
            cv.notify_all();
            std::vector<int> keys(0);
            keys.resize(workers.size());
            for (const auto &[key, _] : workers)
                keys.emplace_back(key);
            for (int key : keys)
                workers[key].join();
        }
        break;
        case ROLE::STUB:
            close(health_check_sock);
            /* kill child process */
            if (fork_state && kill(fork_state, SIGTERM) == -1 && errno != ESRCH)
                LOG_LOCATION("failed to kill runtime");
            break;
        }
    }
    void SetRuntime(int child_pid);
    void Serve(int min_workers, int sync_fd);
    int WaitRuntime(int sync_fd);
    int HookTerminate(int epoll_fd);
    void HandleLoadLib(int fd);
    void HandleInvokeFunc(int fd);
    void HandlePullPage(int fd);
    std::string ReadMsg(int fd);
    std::string AtomicReadMsg(int fd);
    void WriteMsg(int fd, const google::protobuf::Message *msg);
    void AtomicWriteMsg(int fd, const google::protobuf::Message *msg);
    int ConnectRuntimeSocket();
    std::string DispatchHandler(int fd, const google::protobuf::Message *request, REQUEST_CODE request_code);
    std::mutex &GetFDMutex();
    PageManager page_manager;
    template <typename Enum>
    constexpr static auto to_underlying(Enum e) noexcept
    {
        return static_cast<std::underlying_type_t<Enum>>(e);
    }
    const static std::map<StatusCode, std::string> StatusDescription;

private:
    enum class RUNTIME_STATE
    {
        CREATED,
        RUNNING,
        TERMINATING
    };
    constexpr static const int handler_table_size = static_cast<int>(REQUEST_CODE::RESERVED);
    int timeout_ms;
    std::string client_id;
    std::list<void *> sym_handler;
    std::map<uint64_t, std::string> addr2sym;
    const int page_size = getpagesize();
    int health_check_sock;
    int epoll_fd;
    int fork_state;
    std::mutex runtime_mtx;
    ROLE role;
    int listen_request_fd = -1;
    std::condition_variable cv;
    std::queue<int> job_queue;
    int active_workers;
    RUNTIME_STATE state;
    std::map<int, std::thread> workers;
    std::string sun_path;
    std::array<void (IsolatedRuntime::*)(int), handler_table_size> handler_table;
    int min_workers;
    void handle_request(int thread_key);
    std::mutex atomic_fd_mtx;
    int notify_listen(int sync_fd);
    std::mutex resp_id_mtx;
    uint64_t resp_id = 0;
};

class X64SigRPCService final : public SigRPC::Service
{
public:
    /* this constructor throws exception */
    X64SigRPCService(int min_runtime_workers, std::string sock_dir)
    {
        {
            struct sigaction sa = {0};
            sa.sa_handler = SIG_IGN;
            if (sigaction(SIGPIPE, &sa, NULL) == -1)
            {
                LOG_LOCATION("sigaction failed");
                throw std::runtime_error("sigaction failed");
            }
        }
        {
            struct sigaction sa = {0};
            sa.sa_handler = SIG_IGN;
            sa.sa_flags = SA_NOCLDWAIT;
            if (sigaction(SIGCHLD, &sa, NULL) == -1)
            {
                LOG_LOCATION("sigaction failed");
                throw std::runtime_error("sigaction failed");
            }
        }
        this->min_runtime_workers = min_runtime_workers;
        this->sock_dir = sock_dir;
        epoll_fd = epoll_create1(0);
        if (epoll_fd == -1)
        {
            LOG_LOCATION("epoll_create1 failed");
            throw std::runtime_error("epoll_create1 failed");
        }
        std::thread(&X64SigRPCService::run_gc, this).detach();
    }
    Status LoadLib(ServerContext *context, const LoadLibMsg *request, LoadLibMsg *response) override;
    Status InvokeFunc(ServerContext *context, grpc::ServerReaderWriter<InvokeFuncMsg, InvokeFuncMsg> *stream);
    Status PullPage(ServerContext *context, const PullPageMsg *request, PullPageMsg *response) override;

private:
    std::map<std::string, std::pair<int, std::shared_ptr<IsolatedRuntime>>> runtimes;
    std::map<int, std::string> fd2runtime;
    std::shared_mutex runtimes_map_mtx;
    int min_runtime_workers;
    std::string sock_dir;
    int epoll_fd; /* to monitor runtime's socket */
    void run_gc();
};

#endif //(RUNTIME_H)
