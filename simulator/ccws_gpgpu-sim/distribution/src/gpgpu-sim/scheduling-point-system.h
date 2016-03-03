#ifndef __SCHEDULING_POINT_SYSTEM_H__
#define __SCHEDULING_POINT_SYSTEM_H__

#include <vector>
#include <set>
#include "gpu-cache.h"
#include <zlib.h>

enum scheduling_state
{
    SCHED_STATE_NORMAL = 0,
    SCHED_STATE_LOAD_EXCLUSIVE_LIST,
    NUM_SCHED_STATE
};

enum scheduling_youngest_definition
{
    SCHED_HIGHEST_DYNAMID = 0,
    SCHED_LEAST_INST_ISS,
    NUM_SCHED_YOUNGEST_DEFINITION
};

class warp_inst_t;
class shader_core_stats;
class base_shd_warp_t;

typedef std::vector< std::pair< unsigned, int > > VectWarpPointPairs;

class scheduling_point_system_stats
{
public:
    scheduling_point_system_stats(int num_warps)
        : m_vc_hits_per_warp( num_warps, 0 ),
          m_own_hits_per_warp( num_warps, 0 ),
          m_other_hits_per_warp( num_warps, 0 ),
          m_miss_per_warp( num_warps, 0 ),
          m_max_score_per_warp( num_warps, 0 ),
          m_percent_cutoff( num_warps, 0 ),
          m_sum_of_exclusive_warps( 0 ),
          m_sum_normal_active( 0 ),
          m_num_warps( num_warps ),
          m_vc_hit_sum( 0 ),
          m_vc_hit_sum_total( 0 ),
          m_own_hit_sum( 0 ),
          m_own_hit_sum_total( 0 ),
          m_other_hit_sum( 0 ),
          m_other_hit_sum_total( 0 ),
          m_recent_dynamic_cap( 0 ),
          m_max_dynamic_cap( 0 )
          {}

    void visualizer_print( gzFile visualizer_file ) const;
    void print( const std::vector<unsigned>& instr_issued_vec ) const;
    void event_vc_hit(unsigned warp_num)
    {
        ++m_vc_hits_per_warp[warp_num];
        ++m_vc_hit_sum;
    }
    void event_cache_access( cache_request_status status, unsigned warp_num, bool is_own )
    {
        if ( HIT == status ) {
            if (is_own) {
                ++m_own_hits_per_warp[warp_num];
                ++m_own_hit_sum;
            } else {
                ++m_other_hit_sum;
                ++m_other_hits_per_warp[warp_num];
            }
        } else if ( MISS == status ) {
            ++m_miss_per_warp[warp_num];
        }
    }
    void event_new_score( unsigned warp_num, int new_score )
    {
        if (new_score > m_max_score_per_warp[warp_num]) {
            m_max_score_per_warp[warp_num] = new_score;
        }
    }
    void event_clear_exclusive_list()
    {
        std::vector<unsigned>::iterator it = m_percent_cutoff.begin();
        while (it != m_percent_cutoff.end()) {
            *it = 0;
            ++it;
        }
    }
    void event_warp_added_to_exclusive( unsigned warp_num, int points, int cutoff_threshold )
    {
        assert( cutoff_threshold > 0 );
        m_percent_cutoff[warp_num] = points * 100 / cutoff_threshold;
    }
    void event_cycle( scheduling_state current_state, unsigned num_exclusive_warps, unsigned num_active_warps )
    {
        if ( SCHED_STATE_LOAD_EXCLUSIVE_LIST == current_state ) {
            m_sum_of_exclusive_warps += num_exclusive_warps;
        } else if ( SCHED_STATE_NORMAL == current_state ) {
            m_sum_normal_active += num_active_warps;
        }
        ++m_cycles_in_state[ current_state ];
    }

    void event_dynamic_warp_cap_used( unsigned cap )
    {
        m_recent_dynamic_cap = cap;
        if ( cap > m_max_dynamic_cap ) {
            m_max_dynamic_cap = cap;
        }
    }


    float get_vc_hit_cache_hit_ratio() const
    {
        return (float)m_vc_hit_sum/(m_own_hit_sum+m_other_hit_sum + m_vc_hit_sum);
    }

    void event_warp_exit( unsigned warp_id )
    {
        m_vc_hit_sum -= m_vc_hits_per_warp[warp_id];
        m_own_hit_sum -= m_own_hits_per_warp[warp_id];
        m_other_hit_sum -= m_other_hits_per_warp[warp_id];
        m_vc_hit_sum_total += m_vc_hits_per_warp[warp_id];
        m_own_hit_sum_total += m_own_hits_per_warp[warp_id];
        m_other_hit_sum_total += m_other_hits_per_warp[warp_id];
        m_vc_hits_per_warp[warp_id] = 0;
        m_own_hits_per_warp[warp_id] = 0;
        m_other_hits_per_warp[warp_id] = 0;
        m_miss_per_warp[warp_id] = 0;
        m_max_score_per_warp[warp_id] = 0;
        m_percent_cutoff[warp_id] = 0;
    }

    void event_dynamic_vc_hit_pts( int points )
    {
        m_recent_vc_hit_points = points;
        if (points > m_max_vc_hit_point) {
            m_max_vc_hit_point = points;
        }
    }

    unsigned get_vc_hit_sum() const
    {
        return m_vc_hit_sum;
    }
private:
    std::vector<unsigned> m_vc_hits_per_warp;
    std::vector<unsigned> m_own_hits_per_warp;
    std::vector<unsigned> m_other_hits_per_warp;
    std::vector<unsigned> m_miss_per_warp;
    std::vector<int> m_max_score_per_warp;
    std::vector<unsigned> m_percent_cutoff;
    unsigned m_sum_of_exclusive_warps;
    unsigned m_sum_normal_active;
    unsigned m_cycles_in_state[NUM_SCHED_STATE];
    unsigned m_num_warps;
    unsigned m_vc_hit_sum;
    unsigned m_vc_hit_sum_total;
    unsigned m_own_hit_sum;
    unsigned m_own_hit_sum_total;
    unsigned m_other_hit_sum;
    unsigned m_other_hit_sum_total;
    unsigned m_recent_dynamic_cap;
    unsigned m_max_dynamic_cap;
    int m_recent_vc_hit_points;
    int m_max_vc_hit_point;
};

class scheduling_point_system_config {
public:
    scheduling_point_system_config() 
        : m_points_per_vc_hit(0),
          m_points_per_cache_miss(0),
          m_points_per_cache_own_hit(0),
          m_points_per_cache_other_hit(0),
          m_points_per_cycle(0),
          m_global_point_cutoff(1),
          m_youngest_definition(NUM_SCHED_YOUNGEST_DEFINITION),
          m_base_points_per_warp(0),
          m_warp_percent_cap(0),
          m_k_vc_add(2),
          m_k_wsc(10),
          m_base_vc_points(50)
    {
    }

    void init();

    int m_points_per_vc_hit;
    int m_points_per_cache_miss;
    int m_points_per_cache_own_hit;
    int m_points_per_cache_other_hit;
    int m_points_per_cycle;
    int m_global_point_cutoff;
    // Currently not used
    //-----------------------------
    scheduling_youngest_definition m_youngest_definition;
    //-----------------------------
    int m_base_points_per_warp;
    int m_warp_percent_cap;
    int m_k_vc_add;
    int m_k_wsc;
    int m_base_vc_points;


    char* m_config_string;
};

class scheduling_point_system
{
public:
    scheduling_point_system(
        int num_warps, 
        const scheduling_point_system_config* point_system,
        const shader_core_stats* parent_core_stats,
        const std::vector<base_shd_warp_t*>& shader_warps,
        unsigned parent_core_id);

    void vc_hit(int warp_num);
    void cache_access(unsigned warp_id, cache_request_status status, bool own_hit);
    void cycle();
    void print() const;
    void visualizer_print( gzFile vis_file ) const;

    scheduling_state get_state() const {return m_state;}
    bool is_in_exclusive_list(unsigned warp_id)
    {
        return m_exclusive_load_list.find(warp_id)
            != m_exclusive_load_list.end();
    }
    void inform_warp_exit(unsigned warp_id);
    void signal_inst_issue( unsigned warp_id );

    bool operator()(const base_shd_warp_t* lhs, const base_shd_warp_t* rhs);

private:
    void add_youngest_warp_ids(std::set<unsigned>& exclusive_list, unsigned num_youngest) const;
    int evaluate_exclusive_load_list();
    void increment_score(int warp_num, int points);

    int m_num_warps;
    VectWarpPointPairs m_points_per_warp;
    std::vector< unsigned > m_issued_inst_per_warp;
    unsigned m_issued_inst_total;

    std::set<unsigned> m_exclusive_load_list;
    const scheduling_point_system_config* m_config;
    const shader_core_stats* m_core_stats;
    const std::vector<base_shd_warp_t*>& m_shader_warps;
    unsigned m_sid;
    int m_last_global_score;
    int m_global_cutoff;
    mutable scheduling_point_system_stats m_stats;
    scheduling_state m_state;
};

#endif
