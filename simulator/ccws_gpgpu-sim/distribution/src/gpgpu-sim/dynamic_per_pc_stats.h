// Copyright (c) 2009-2011, Timothy G. Rogers
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __DYNAMIC_PER_PC_STATS_H__
#define __DYNAMIC_PER_PC_STATS_H__

#include "../abstract_hardware_model.h"
#include "shader.h"

#include <string>

struct dynamic_per_pc_stats_config {
    dynamic_per_pc_stats_config()
        : enable_warp_issue_insn( false ),
          enable_warp_l1_data_cache_access( false ),
          enable_warp_l1_data_cache_hit( false ) {}

    static const char* WARP_ISSUE_INSN;
    static const char* WARP_CACHE_HIT_STATS;
    static const char* WARP_CACHE_ACCESS_STATS;

    void init() {
        std::string config( config_string );
        if ( config.find( WARP_ISSUE_INSN ) != std::string::npos ) {
            enable_warp_issue_insn = true;
        }
        if ( config.find( WARP_CACHE_HIT_STATS ) != std::string::npos ) {
            enable_warp_l1_data_cache_hit = true;
        }
        if ( config.find( WARP_CACHE_ACCESS_STATS ) != std::string::npos ) {
            enable_warp_l1_data_cache_access = true;
        }
    }

    bool enable_warp_issue_insn;
    bool enable_warp_l1_data_cache_access;
    bool enable_warp_l1_data_cache_hit;
    char* config_string;
};

class dynamic_per_pc_stats {
public:

    dynamic_per_pc_stats( const dynamic_per_pc_stats_config* config, const shader_core_config* shader_config, address_type start_pc, unsigned long long  logging_interval );

    void log_warp_issue_insn( unsigned sid, unsigned warp_id, address_type pc );
    void log_warp_l1_data_cache_access( unsigned warp_id, address_type pc );
    void log_warp_l1_data_cache_hit( unsigned warp_id, address_type pc );

private:
    const dynamic_per_pc_stats_config* m_config;
};

#endif //__DYNAMIC_PER_PC_STATS_H__
