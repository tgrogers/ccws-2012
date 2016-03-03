
#include "scheduling-point-system.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "../abstract_hardware_model.h"
#include "shader.h"

void scheduling_point_system_config::init()
{
    if ( strncmp("none", m_config_string, 4) != 0 ) {
        sscanf( m_config_string, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", 
            &m_points_per_vc_hit,
            &m_points_per_cache_own_hit,
            &m_points_per_cache_other_hit,
            &m_points_per_cache_miss,
            &m_points_per_cycle,
            &m_global_point_cutoff,
            &m_base_points_per_warp,
            &m_warp_percent_cap,
            &m_k_vc_add,
            &m_k_wsc,
            &m_base_vc_points );
    }
}

scheduling_point_system::scheduling_point_system(
    int num_warps,
    const scheduling_point_system_config* config,
    const shader_core_stats* core_stats,
    const std::vector<base_shd_warp_t*>& shader_warps,
    unsigned parent_core_id )
        :m_num_warps(num_warps), m_config(config), 
         m_core_stats(core_stats), m_shader_warps(shader_warps),
         m_sid(parent_core_id), m_last_global_score(0),
         m_global_cutoff(1), m_stats(num_warps),
         m_issued_inst_per_warp(num_warps),
         m_issued_inst_total(0),
         m_state(SCHED_STATE_NORMAL)
{
    m_points_per_warp.resize(m_num_warps);
    for (int i = 0; i < m_num_warps; ++i) {
        m_points_per_warp[i].first = i;
        m_points_per_warp[i].second = m_config->m_base_points_per_warp;
    }
    if ( m_config->m_global_point_cutoff > 0 ) {
        m_global_cutoff = m_config->m_global_point_cutoff;
    } else {
        m_global_cutoff = num_warps * m_config->m_base_points_per_warp;
    }
}

void scheduling_point_system::vc_hit(int warp_num)
{
    m_stats.event_vc_hit( warp_num );
    int points_to_add = m_config->m_points_per_vc_hit;
    if (  m_config->m_points_per_vc_hit < 0 ) {
        points_to_add = m_config->m_base_vc_points 
//                      + m_stats.get_vc_hit_cache_hit_ratio() 
                      + (float)m_stats.get_vc_hit_sum() / m_issued_inst_total 
                        * m_config->m_k_vc_add * m_global_cutoff;
        m_stats.event_dynamic_vc_hit_pts(points_to_add);
    }
    increment_score( warp_num, points_to_add );
}

void scheduling_point_system::cache_access( unsigned warp_id,
                                            cache_request_status status,
                                            bool hit_own )
{
    m_stats.event_cache_access( status, warp_id, hit_own );
    if ( HIT == status) {
        if ( hit_own ) {
            increment_score(warp_id, m_config->m_points_per_cache_own_hit);
        } else {
            increment_score(warp_id, m_config->m_points_per_cache_other_hit);
        }
    } else if ( MISS == status ) {
        increment_score( warp_id, m_config->m_points_per_cache_miss );
    }
}

void scheduling_point_system::cycle()
{
    for (VectWarpPointPairs::iterator it = m_points_per_warp.begin();
         it != m_points_per_warp.end(); ++it) {
        if (it->second > m_config->m_base_points_per_warp) {
            increment_score( it->first, m_config->m_points_per_cycle );
        }
    }
    int num_total_active_warps = evaluate_exclusive_load_list();
    if ( m_config->m_global_point_cutoff < 0 ) {
        m_global_cutoff = std::max( 1, num_total_active_warps * m_config->m_base_points_per_warp);
    }
    m_stats.event_cycle(m_state, m_exclusive_load_list.size(), num_total_active_warps);
}

void scheduling_point_system::inform_warp_exit(unsigned warp_id) {
    if (m_last_global_score >= m_global_cutoff
        && m_exclusive_load_list.find(warp_id) != m_exclusive_load_list.end()) {
        assert(SCHED_STATE_LOAD_EXCLUSIVE_LIST == m_state);
        m_exclusive_load_list.erase(warp_id);
        m_points_per_warp[warp_id].second = m_config->m_base_points_per_warp;
    }
    m_points_per_warp[warp_id].second = 0;
    m_issued_inst_total -= m_issued_inst_per_warp[warp_id];
    m_issued_inst_per_warp[warp_id] = 0;
    m_stats.event_warp_exit(warp_id);
}

bool younger_warp_dyn_id(const base_shd_warp_t* lhs, const base_shd_warp_t* rhs)
{
    return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id();
}

bool scheduling_point_system::operator()(const base_shd_warp_t* lhs, const base_shd_warp_t* rhs)
{
    return m_core_stats->m_shader_dynamic_warp_issue_distro[m_sid][lhs->get_dynamic_warp_id()] < m_core_stats->m_shader_dynamic_warp_issue_distro[m_sid][rhs->get_dynamic_warp_id()];
}

void scheduling_point_system::add_youngest_warp_ids(std::set<unsigned>& exclusive_list, unsigned num_youngest) const
{
    std::vector<const base_shd_warp_t*> sorted_list;
    std::vector<base_shd_warp_t*>::const_iterator iter = m_shader_warps.begin();
    while (iter != m_shader_warps.end()) {
        if (!(*iter)->done_exit()) {
            sorted_list.push_back(*iter);
        }
        ++iter;
    }
   
    if (SCHED_HIGHEST_DYNAMID == m_config->m_youngest_definition) {
        std::sort(sorted_list.begin(), sorted_list.end(), younger_warp_dyn_id);
    } else if (SCHED_LEAST_INST_ISS == m_config->m_youngest_definition) {
        std::sort(sorted_list.begin(), sorted_list.end(), *this);
    } else {
        fprintf(stderr, "Unkown youngest warp definition\n");
        abort();
    }
    
    std::vector<const base_shd_warp_t*>::const_iterator iter2 = sorted_list.begin();
    unsigned num_added = 0;
    while (iter2 != sorted_list.end() && num_added < num_youngest) {
        exclusive_list.insert((*iter2)->get_warp_id());
        ++num_added;
        ++iter2;
    }
}

bool highest_second( const std::pair<unsigned, int>& lhs, 
                     const std::pair<unsigned, int>& rhs )
{
    if ( lhs.second == rhs.second ) {
        return lhs.first < rhs.first;
    } else {
        return lhs.second > rhs.second;
    }
}

int scheduling_point_system::evaluate_exclusive_load_list()
{
    m_exclusive_load_list.clear();
    m_stats.event_clear_exclusive_list();

    std::set<unsigned> active_warp_set;
    for ( std::vector<base_shd_warp_t*>::const_iterator it
            =  m_shader_warps.begin(); it != m_shader_warps.end();
            ++it ) {
        if (!(*it)->functional_done()) {
            active_warp_set.insert( (*it)->get_warp_id() );
        }
    }

    int global_score = 0;
    VectWarpPointPairs sorted_warp_points = m_points_per_warp;
    std::sort( sorted_warp_points.begin(), sorted_warp_points.end(), highest_second );
    bool was_limited = false;
    for (VectWarpPointPairs::iterator iter = sorted_warp_points.begin();
         iter != sorted_warp_points.end(); ++iter) {
        if (active_warp_set.find(iter->first) != active_warp_set.end()) {
            global_score += iter->second;
            m_exclusive_load_list.insert(iter->first);
            m_stats.event_warp_added_to_exclusive( iter->first,
                                                iter->second,
                                                m_global_cutoff );
            if (global_score >= m_global_cutoff) {
                was_limited = true;
                break;
            }
        }
    }
    m_last_global_score = global_score;   

    if (was_limited)  {
        m_state = SCHED_STATE_LOAD_EXCLUSIVE_LIST;
    } else {
        m_state = SCHED_STATE_NORMAL;
    }
    return active_warp_set.size();
}

void scheduling_point_system::increment_score(int warp_num, int points)
{
    m_points_per_warp[warp_num].second += points;
    if (m_points_per_warp[warp_num].second <= m_config->m_base_points_per_warp) {
        m_points_per_warp[warp_num].second = m_config->m_base_points_per_warp;
    } else if ( m_config->m_warp_percent_cap > 0 
                && m_points_per_warp[warp_num].second * 100 / m_global_cutoff 
                    > m_config->m_warp_percent_cap ) {
        m_points_per_warp[warp_num].second = m_config->m_warp_percent_cap * m_global_cutoff / 100;
    } else if ( m_config->m_warp_percent_cap == -1 ) {
        const int dynamic_cap 
//            = m_stats.get_vc_hit_cache_hit_ratio() * m_config->m_k_wsc  * m_global_cutoff;
            = (float)m_stats.get_vc_hit_sum() / m_issued_inst_total * m_config->m_k_wsc  * m_global_cutoff;
        if (m_points_per_warp[warp_num].second > dynamic_cap) {
            m_stats.event_dynamic_warp_cap_used(dynamic_cap);
            m_points_per_warp[warp_num].second = dynamic_cap;
        }
    }
    m_stats.event_new_score( warp_num, m_points_per_warp[warp_num].second );
}

void scheduling_point_system::signal_inst_issue(unsigned warp_id)
{
    ++m_issued_inst_per_warp[warp_id];
    ++m_issued_inst_total;
}

void scheduling_point_system::print() const
{
    printf("SPS_global_cutoff = %u\n", m_global_cutoff);
    m_stats.print( m_issued_inst_per_warp );
}

void scheduling_point_system_stats::print( const std::vector<unsigned>& instr_issued_vec ) const
{
    unsigned total_cycles = 0;
    for ( unsigned i = 0; i < NUM_SCHED_STATE; ++i ) {
        total_cycles += m_cycles_in_state[i];
    }
    printf("SPS_normal_cycles=%d\n", m_cycles_in_state[SCHED_STATE_NORMAL]);
    printf("SPS_percent_normal_cycles=%.2f\n",
           (float)m_cycles_in_state[SCHED_STATE_NORMAL] / (float)total_cycles);
    printf("SPS_exclusive_cycles=%d\n",
           m_cycles_in_state[SCHED_STATE_LOAD_EXCLUSIVE_LIST]);
    printf("SPS_percent_exclusive_cycles=%.2f\n",
           (float)m_cycles_in_state[SCHED_STATE_LOAD_EXCLUSIVE_LIST]
           / (float)total_cycles);
    printf("SPS_avg_warps_exclusive=%.2f\n", 
           (float)m_sum_of_exclusive_warps
           / (float) m_cycles_in_state[SCHED_STATE_LOAD_EXCLUSIVE_LIST]);
    printf("SPS_normalized_avg_exclusive_warps=%.2f\n",
           (float)(m_sum_of_exclusive_warps + m_sum_normal_active)/ (float) total_cycles);
    printf("SPS_vc_hit_sum_total=%u\n", m_vc_hit_sum_total);
    printf("SPS_own_hit_sum_total=%u\n", m_own_hit_sum_total);
    printf("SPS_other_hit_sum_total=%u\n", m_other_hit_sum_total);
    printf("SPS_vc_hit_sum=%u\n", m_vc_hit_sum);
    printf("SPS_own_hit_sum=%u\n", m_own_hit_sum);
    printf("SPS_vc_hit_cache_hit_ratio=%.2f\n", (float)m_vc_hit_sum/(m_other_hit_sum + m_own_hit_sum));
    printf("SPS_other_hit_sum=%u\n", m_other_hit_sum);
    printf("SPS_current_dynamic_cap=%u\n", m_recent_dynamic_cap);
    printf("SPS_max_dynamic_cap=%u\n", m_max_dynamic_cap);
    printf("SPS_recent_vc_hit_points=%u\n", m_recent_vc_hit_points);
    printf("SPS_max_vc_hit_point=%u\n", m_max_vc_hit_point);
    for ( unsigned warp_id = 0; warp_id < m_num_warps; ++warp_id ) {
        printf("Warp %d - MAX SCORE %d - VC hits %u - Own Hits %u - Other Hits %u - Misses %u VC/Own Ratio %.2f - Instr Issued %u - Percent Cutoff %u\n",
                warp_id,
                m_max_score_per_warp[warp_id],
                m_vc_hits_per_warp[warp_id],
                m_own_hits_per_warp[warp_id],
                m_other_hits_per_warp[warp_id],
                m_miss_per_warp[warp_id],
                (float)m_vc_hits_per_warp[warp_id]/m_own_hits_per_warp[warp_id],
                instr_issued_vec[warp_id],
                m_percent_cutoff[warp_id]
                );
    }
}

void scheduling_point_system::visualizer_print( gzFile visualizer_file ) const
{
    m_stats.visualizer_print( visualizer_file );
}

void scheduling_point_system_stats::visualizer_print( gzFile visualizer_file ) const
{
    gzprintf(visualizer_file, "DL1VCHitBreakdown:");
    gzprintf(visualizer_file,"\n");

}
