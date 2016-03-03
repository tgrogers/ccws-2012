#ifndef _PRIORITY_CACHE_H_
#define _PRIORITY_CACHE_H_

#include "gpu-cache.h"
#include "scheduling-point-system.h"

extern int VC_PER_WARP;

struct priority_cache_block : public cache_block_t {
    priority_cache_block( unsigned line_size )
        : cache_block_t( line_size ), m_cycles_protected( 0 ), m_accesses_protected( 0 ), m_source_pc( 0 ),
          m_source_warp(0xDEADBEEF) {}
    virtual ~priority_cache_block(){}

    virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time ) {
        cache_block_t::allocate( tag, block_addr, time );
        // TODO tgrogers - the line protection should happen here...
    }

    virtual void fill( unsigned time )
    {
        cache_block_t::fill( time );
    }

    unsigned m_cycles_protected;
    unsigned m_accesses_protected;
    std::map< unsigned, unsigned > m_warps_protecting_line;
    new_addr_type m_source_pc;
    unsigned m_source_warp;
};

struct victim_cache_block : public cache_block_t {
    victim_cache_block( unsigned line_size )
        : cache_block_t( line_size ), m_source_pc( 0 ) {}
    virtual ~victim_cache_block(){}

    new_addr_type m_source_pc;
};

class victim_tag_array : public base_tag_array {
public:
    victim_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump )
        : base_tag_array( config,core_id,type_id, enable_access_stream_dump ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new victim_cache_block( config.get_line_sz() );
        }
    }

    virtual ~victim_tag_array(){
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }
};


extern unsigned PROTECTED_CYCLES;
extern unsigned PROTECTED_ACCESES;
enum protection_type {
    CYCLES_PROTECTION_TYPE = 0,
    ACCESSES_PROTECTION_TYPE
};
class priority_tag_array : public base_tag_array {
public:
    priority_tag_array( const cache_config &config, int core_id, int type_id, unsigned num_warps, unsigned so_factor, bool enable_access_stream_dump )
        : base_tag_array( config,core_id,type_id, enable_access_stream_dump ), m_num_protected_lines( 0 ), m_stall_override_factor( so_factor ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new priority_cache_block( config.get_line_sz() );
        }
        m_lines_protected_per_warp.resize( num_warps );
        for ( unsigned i = 0; i < num_warps; ++i ) {
            m_lines_protected_per_warp[ i ].first = i;
        }
    }

    virtual ~priority_tag_array(){
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }

    void cycle();

    void remove_protected_lines( unsigned warp_id ) {
        //printf( "removing protection for warp=%u\n", warp_id );
        for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
            if ( (*it)->m_warps_protecting_line.find( warp_id ) != (*it)->m_warps_protecting_line.end() ) {
                (*it)->m_warps_protecting_line.erase( warp_id );
            }
        }
    }

    void print_protected_info( FILE* fout ) {
        unsigned num_lines_protected = 0;
        if ( get_protection_type() == CYCLES_PROTECTION_TYPE ) {
            for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
                if ( ( !hack_use_warp_prot_list && (*it)->m_cycles_protected > 0 ) || ( hack_use_warp_prot_list && (*it)->m_warps_protecting_line.size() > 0 ) ) {
                    ++num_lines_protected;
                }
            }
        } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE ) {
            for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
                if ( (*it)->m_accesses_protected > 0 ) {
                    ++num_lines_protected;
                }
            }
        } else {
            abort();
        }

        fprintf( fout, "cache_protected_lines=%u\ncache_percent_lines_protected=%f\n"
                , num_lines_protected, ( float ) num_lines_protected / ( float ) m_lines.size() );

        if ( hack_use_warp_prot_list ) {
            fprintf( fout, "warps protecting each line:\n" );
            for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
                fprintf( fout, "%u, ", (*(*it)->m_warps_protecting_line.begin()).first );
            }
            fprintf( fout, "\n" );
        }
    }

    //virtual enum cache_request_status probe( new_addr_type addr, unsigned &idx ) const;

    cache_request_status probe_special( new_addr_type addr, unsigned &idx, unsigned& num_protected ) const;

    unsigned get_num_protected_lines() const { return m_num_protected_lines; }
    void protect_line( unsigned cache_index, unsigned warp_id );
    bool should_whitelist_warp( unsigned warp_id ) const;

    protection_type get_protection_type() const {
        // One or the other... or neither
        assert( ( PROTECTED_CYCLES > 0 ) ^ ( PROTECTED_ACCESES > 0 )
                || (PROTECTED_CYCLES == 0 && PROTECTED_ACCESES == 0) );
        return PROTECTED_CYCLES > 0 ? CYCLES_PROTECTION_TYPE : ACCESSES_PROTECTION_TYPE;
    }

    virtual enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ) {
        for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
            if ( (*it)->m_accesses_protected > 0 ) {
                assert( get_protection_type() == ACCESSES_PROTECTION_TYPE );
                --(*it)->m_accesses_protected;
                if ( (*it)->m_accesses_protected == 0 ) {
                    --m_num_protected_lines;
                    assert( (*it)->m_warps_protecting_line.size() == 1 );
                    --m_lines_protected_per_warp[ ( ( *it )->m_warps_protecting_line.begin() )->first ].second;
                    (*it)->m_warps_protecting_line.clear();
                }
            }
        }
        assert( m_num_protected_lines <= m_lines.size() );
        cache_request_status result = base_tag_array::access( addr, time, idx, wb, evicted, NULL );

        // This is a hack to clear protection on lines that do not deserve it....
        if ( MISS == result ) {
            if ( get_protection_type() == ACCESSES_PROTECTION_TYPE ) {
                assert( m_lines[idx]->m_accesses_protected != PROTECTED_ACCESES );
            }

            if ( get_protection_type() == CYCLES_PROTECTION_TYPE
                    && m_lines[idx]->m_cycles_protected != PROTECTED_CYCLES
                    && m_lines[idx]->m_cycles_protected > 0 ) {
                m_lines[idx]->m_cycles_protected = 0;
                --m_num_protected_lines;
            } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE
                    && m_lines[idx]->m_accesses_protected != ( PROTECTED_ACCESES - 1)
                    && m_lines[idx]->m_accesses_protected > 0 ) {
                m_lines[idx]->m_accesses_protected = 0;
                --m_num_protected_lines;
            }
        }
        return result;

    }

private:
    std::vector< priority_cache_block * > m_lines;
    unsigned m_num_protected_lines;
    unsigned m_stall_override_factor;

    typedef std::pair< unsigned, unsigned > count_warp_id_pair;
    std::vector< count_warp_id_pair > m_lines_protected_per_warp;
};

class high_locality_protected_cache : public evict_on_write_cache {
public:
    high_locality_protected_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status, const high_locality_cache_config& high_loc_conf, unsigned num_warps,
                const cache_config &victim_cache_config, bool enable_access_dump )
    : evict_on_write_cache(name,config,core_id,type_id,memport,mfcreator,status,enable_access_dump), m_high_loc_config( high_loc_conf ) {
        m_dynamic_detection_pcs_to_protect_per_warp.resize(num_warps);
        evict_on_write_cache::m_tag_array = m_tag_array = new priority_tag_array( config,core_id,type_id, num_warps, high_loc_conf.get_so_factor(), enable_access_dump );
        if ( !victim_cache_config.disabled() ) {
            if ( VC_PER_WARP ) {
                for (unsigned i = 0; i < num_warps; ++i ) {
                    m_victim_tag_arrays.push_back(new victim_tag_array( victim_cache_config, core_id, type_id, enable_access_dump ));
                }

            } else {
                m_victim_tag_array = new victim_tag_array( victim_cache_config, core_id, type_id, enable_access_dump );
            }
        }
    }

    virtual ~high_locality_protected_cache() {
        delete m_tag_array;
    }

    virtual void cycle();

    enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events, warp_inst_t &inst );

    void remove_protected_lines( unsigned warp_id ) {
        m_tag_array->remove_protected_lines( warp_id );
    }


    void print_victim_cache_results() const;

    virtual void print(FILE *fp, unsigned &accesses, unsigned &misses);

    virtual cache_request_status probe( new_addr_type addr ) const {
        unsigned dummy;
        unsigned num_pro = 0;
        cache_request_status probe_res = m_tag_array->probe_special( addr, dummy, num_pro );
        //return probe_res;
        return num_pro > 0 && RESERVATION_FAIL == probe_res ? FAIL_PROTECTION : probe_res;
        //return m_tag_array->probe( addr, dummy );
    }

    bool handle_victim_cache_probe( new_addr_type block_addr, const warp_inst_t &inst );

    bool should_whitelist_warp( unsigned warp_id ) const;

    void set_point_system(scheduling_point_system* system)
    {
        m_point_system = system;
    }

    virtual void flush();

private:
    priority_tag_array* m_tag_array;
    const high_locality_cache_config& m_high_loc_config;

    // Constructs used for idealized high locality detection
    struct victim_cache_entry {
        new_addr_type source_pc;
        unsigned warp_id;
        unsigned cycles_until_detection;

        victim_cache_entry() :
            source_pc( 0 ), warp_id( 0xFFFFFFFF ), cycles_until_detection( 0 ){}
    };
    std::list< new_addr_type > m_victim_cache_deletion_order;
    std::map< new_addr_type, victim_cache_entry > m_victim_cache_hash;

    // Realistic victim cache
    victim_tag_array* m_victim_tag_array;
    std::vector<victim_tag_array*> m_victim_tag_arrays; // for when we have one per warp
    std::vector< std::map< new_addr_type, unsigned > > m_dynamic_detection_pcs_to_protect_per_warp;

    std::map< new_addr_type, unsigned > m_dynamic_detection_pcs_to_protect;
    std::list< new_addr_type > m_reduced_list_to_protect;

    scheduling_point_system* m_point_system;

    void locality_detection_handle_eviction( new_addr_type block_addr, new_addr_type pc, unsigned warp_id );
    void set_limited_protected_lines();
    bool does_have_dynamic_high_locality( new_addr_type pc ) const;
};


#endif
