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

#include "dynamic_per_pc_stats.h"
#include "stat-tool.h"

const char* dynamic_per_pc_stats_config::WARP_ISSUE_INSN = "warp_issue_insn";
const char* dynamic_per_pc_stats_config::WARP_CACHE_HIT_STATS = "warp_cache_hits";
const char* dynamic_per_pc_stats_config::WARP_CACHE_ACCESS_STATS = "warp_cache_accesses";

dynamic_per_pc_stats::dynamic_per_pc_stats( const dynamic_per_pc_stats_config* config, const shader_core_config* shader_config, address_type start_pc, unsigned long long  logging_interval )
        : m_config( config ) {
        if ( m_config->enable_warp_issue_insn ) {
            for ( unsigned i = 0; i < shader_config->n_simt_clusters * shader_config->n_simt_cores_per_cluster; ++i ) {
                char str_buff[ 255 ];
                str_buff[ sizeof( str_buff ) - 1 ] = '\0';
                snprintf( str_buff, sizeof( str_buff ) - 1, "_shader_%02d_", i );
                std::string issuer_name = dynamic_per_pc_stats_config::WARP_ISSUE_INSN;
                issuer_name += str_buff;
                create_thread_CFlogger( issuer_name, shader_config->max_warps_per_shader, shader_config->max_warps_per_shader, start_pc, logging_interval );
            }
        }
        if ( m_config->enable_warp_l1_data_cache_access ) {
            create_thread_CFlogger( dynamic_per_pc_stats_config::WARP_CACHE_ACCESS_STATS, shader_config->max_warps_per_shader, shader_config->max_warps_per_shader, start_pc, logging_interval );
        }
        if ( m_config->enable_warp_l1_data_cache_hit ) {
            create_thread_CFlogger( dynamic_per_pc_stats_config::WARP_CACHE_HIT_STATS, shader_config->max_warps_per_shader, shader_config->max_warps_per_shader, start_pc, logging_interval );
        }
    }

void dynamic_per_pc_stats::log_warp_issue_insn( unsigned sid, unsigned warp_id, address_type pc ) {
    if ( m_config->enable_warp_issue_insn ) {
        char str_buff[ 255 ];
        str_buff[ sizeof( str_buff ) - 1 ] = '\0';
        snprintf( str_buff, sizeof( str_buff ) - 1, "_shader_%02d_", sid );
        std::string issuer_name = dynamic_per_pc_stats_config::WARP_ISSUE_INSN;
        issuer_name += str_buff;
        cflog_update_thread_pc( issuer_name, warp_id, warp_id, pc );
    }
}

void dynamic_per_pc_stats::log_warp_l1_data_cache_access( unsigned warp_id, address_type pc ) {
    if ( m_config->enable_warp_l1_data_cache_access ) {
        cflog_update_thread_pc( dynamic_per_pc_stats_config::WARP_CACHE_ACCESS_STATS, warp_id, warp_id, pc );
    }
}

void dynamic_per_pc_stats::log_warp_l1_data_cache_hit( unsigned warp_id, address_type pc ) {
    if ( m_config->enable_warp_l1_data_cache_hit ) {
        cflog_update_thread_pc( dynamic_per_pc_stats_config::WARP_CACHE_HIT_STATS, warp_id, warp_id, pc );
    }
}
