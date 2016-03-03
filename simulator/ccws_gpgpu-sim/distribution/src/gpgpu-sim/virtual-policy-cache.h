#ifndef _VIRTUAL_POLICY_CACHE_H_
#define _VIRTUAL_POLICY_CACHE_H_

#include "gpu-cache.h"

struct virtual_policy_cache_block_t : public cache_block_t {
    virtual_policy_cache_block_t( unsigned line_size )
        : cache_block_t( line_size ) { }
    virtual ~virtual_policy_cache_block_t() {}
    virtual void fill( unsigned time )
    {
        // tgrogers - changing the logic here for the purposes of our vm remapping system that may invalidate lines that
        //            have not yet been filled.  In which case, lets just ignore these fills...
        //assert( m_status == RESERVED );
        if ( m_status == RESERVED ) {
            m_status=VALID;
            m_fill_time=time;
        }
    }

    vm_page_mapping_payload m_policy;
};


class virtual_policy_tag_array : public base_tag_array {
public:
    virtual_policy_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump )
        : base_tag_array( config,core_id,type_id,enable_access_stream_dump ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new virtual_policy_cache_block_t( config.get_line_sz() );
        }
    }
    virtual ~virtual_policy_tag_array(){
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }

    virtual_policy_cache_block_t &get_block(unsigned idx) const { return *m_lines[ idx ]; }

private:
    std::vector< virtual_policy_cache_block_t * > m_lines;
};

class virtual_policy_cache : public evict_on_write_cache {
public:
    virtual_policy_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                    mem_fetch_allocator *mfcreator, enum mem_fetch_status status, bool enable_access_dump )
        : evict_on_write_cache(name,config,core_id,type_id,memport,mfcreator,status, enable_access_dump) {
        evict_on_write_cache::m_tag_array = m_tag_array = new virtual_policy_tag_array( config, core_id, type_id, enable_access_dump );
    }

    virtual ~virtual_policy_cache() {
        delete m_tag_array;
    }

    virtual void fill( mem_fetch *mf, unsigned time ) {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
        if ( mf->get_payload() ) {
            m_tag_array->get_block( e->second.m_cache_index ).m_policy = *(dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() ));
        }
        evict_on_write_cache::fill( mf, time );
    }

    void invalidate_line_if_policy_does_not_match( new_addr_type addr, const vm_page_mapping_payload& configuration ) {
       unsigned cache_index = (unsigned)-1;
       new_addr_type block_addr = m_config.block_addr(addr);
       enum cache_request_status status = m_tag_array->probe( block_addr,cache_index );
       if ( ( HIT == status || HIT_RESERVED == status ) && m_tag_array->get_block(cache_index).m_policy != configuration ) {
          m_tag_array->get_block( cache_index ).m_status = INVALID;
       }
    }

    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );

private:
    virtual_policy_tag_array* m_tag_array;
};

#define MAX_REGION_SIZE 2048

class vm_manager_config : public cache_config {
public:
    enum obj_size_policy {
       OBJ_SIZE_POLICY_TYPE_INVALID = 0,
       OBJ_SIZE_POLICY_TYPE_GCD,
       OBJ_SIZE_POLICY_TYPE_BYTE_LINE_INTERPRETATION,
       OBJ_SIZE_POLICY_TYPE_MAX
    };

    enum update_policy {
        UPDATE_POLICY_USE_REGION_CACHE,
        UPDATE_POLICY_DIRECT_TO_TLB,
        UPDATE_POLICY_MAX
    };

    vm_manager_config() : m_object_size_policy( OBJ_SIZE_POLICY_TYPE_INVALID ) { }

    virtual void init()
    {
        unsigned enabled=0;
        if ( strncmp( m_config_string, "none", 4 ) == 0 ) {
            m_valid = false;
            m_enabled = false;
            return;
        }
        sscanf(m_config_string,"%u:%u:%u:%u", &enabled, &m_threshold, &m_word_size, &m_max_tile_size);
        m_enabled=(enabled!=0);
        m_object_size_policy = m_enabled ? obj_size_policy( enabled ) : OBJ_SIZE_POLICY_TYPE_INVALID;
        assert( m_object_size_policy >= OBJ_SIZE_POLICY_TYPE_INVALID && m_object_size_policy < OBJ_SIZE_POLICY_TYPE_MAX  );
        const char *config = strchr(m_config_string,',');
        assert(config);
        cache_config::init(++config);
        if( region_size() > MAX_REGION_SIZE ) {
            printf("GPGPU-Sim ERROR ** vm_manager_config region size %u larger than MAX_REGION_SIZE (%u)\n", region_size(), MAX_REGION_SIZE );
            abort();
        }

        m_update_policy = UPDATE_POLICY_DIRECT_TO_TLB; //TODO allow this to be runtime configurable
    }

    unsigned region_size() const { return get_line_sz() / m_word_size; }

    unsigned m_threshold;
    unsigned m_word_size;
    unsigned m_max_tile_size; 

    bool m_enabled;

    obj_size_policy m_object_size_policy;
    update_policy m_update_policy;
};

class vm_policy_manager_stats {
public:
    vm_policy_manager_stats()
        : m_num_accesses_with_no_object_size( 0 ),
          m_num_attempted_insertions( 0 ),
          m_num_insertions_that_already_exist( 0 ) {}

    void print( FILE* fp ) const {
        fprintf( fp, "vm_policy_manager_stats::m_num_accesses_with_no_object_size=%u\n",m_num_accesses_with_no_object_size );
        fprintf( fp, "vm_policy_manager_stats::m_num_attempted_insertions=%u\n",m_num_attempted_insertions );
        fprintf( fp, "vm_policy_manager_stats::m_num_insertions_that_already_exist=%u\n",m_num_insertions_that_already_exist );
    }

    void attemtped_policy_event( const vm_page_mapping_payload* best, size_t object_size ) {
        if ( best ) {
            ++m_num_attempted_insertions;
        }

        if ( object_size == 0 ) {
            ++m_num_accesses_with_no_object_size;
        }
    }

    void redundant_policy_event() {
        ++m_num_insertions_that_already_exist;
    }

private:
    unsigned m_num_accesses_with_no_object_size;
    unsigned m_num_attempted_insertions;
    unsigned m_num_insertions_that_already_exist;
};

// All the dynamic determination of page properties is done here
// Jobs:
//  1) Dynamically Determine Object Size through two possible methods:
//      (a) GCD within the addresses in the instruction (POLICY_TYPE_GCD)
//      (b) From a bit vector of accesses to a "region" i.e. a page (POLICY_TYPE_BYTE_LINE_INTERPRETATION) The bit vector is stored in m_data and is sized
//              to the cache dimensions
// Based on which page format we are using it determines the following ( Only one of the following can be true )
//  if VP_TILED - Note, this assumes completely aligned data
//      2) The field size used (currently just passed as a config option, although the byte access line might be used to determine this)
//      3) The tile width to use (just now just tiles the whole page)
//  if VP_HOT_DATA_ALIGNED - Note, this assumes completely aligned data
//      2) The hot bytes in each object - m_obj_size_to_hit_vector stores this for all our different object sizes.
//  if VP_HOT_DATA_UNALIGNED
//      2) The hot bytes in each object - m_obj_size_to_hit_vector stores this for all our different object sizes.
//      3) The start of our "first page", used as the compass for all other accesses to this array
class vm_policy_manager {
public:
    vm_policy_manager( const vm_manager_config &config, 
                       mem_fetch_allocator *mf_allocator, 
                       mem_fetch_interface *icnt, 
                       enum mem_fetch_status status,
                       unsigned core_id,
                       size_t l1_line_size,
                       virtual_page_factory* factory )
        :   m_page_factory( factory ),
            m_config(config),
            m_tags(config,-1,-1,false),
            m_l1_cache_line_size( l1_line_size )

    {
        m_data = new region_entry[ config.get_num_lines() ];
        m_icnt = icnt;
        m_send_status = status;
        m_mf_allocator=mf_allocator;
        m_core_id = core_id;
    }
    void access( warp_inst_t& inst, unsigned time, shader_page_reordering_tlb& tlb );
    vm_page_mapping_payload *cycle();
    void fill( mem_fetch *mf );
    void update_object_access_vector( const warp_inst_t& inst );

    void print( FILE* fp ) const;

private:
    static const size_t INVALID_OBJECT_SIZE = 0xDEADBEEFDEADBEEF;

    struct region_entry {
        region_entry() { m_count=0; m_pending_change=false; }

        unsigned access(unsigned offset)
        {
            m_accessed.set(offset);
            m_count++;
            return m_count;
        }
       
        unsigned m_count;
        std::bitset<MAX_REGION_SIZE> m_accessed;
        bool m_pending_change;
    };

    void print_object_access_info( FILE* fp ) const;

    size_t determine_gcd_obj_size( const warp_inst_t& instr ) const;
    size_t gcd( size_t a, size_t b ) const ;

    virtual_page_factory* m_page_factory;

    // for transpose vm stuff
    const vm_manager_config &m_config;
    tag_array m_tags;
    region_entry *m_data;

    vm_page_mapping_payload select_tiled_mapping( new_addr_type block_addr, size_t obj_size );

    unsigned find_hot_data( boost::dynamic_bitset<>& hotmap, size_t object_size ) const;
    vm_page_mapping_payload select_aligned_hot_mapping( new_addr_type block_addr, size_t obj_size );
    vm_page_mapping_payload select_unaligned_hot_mapping( new_addr_type block_addr, size_t obj_size );

    size_t determine_region_access_vector_obj_size( const struct region_entry &region ) const;
    void update_region_access_vector( warp_inst_t& inst, unsigned time );

    vm_page_mapping_payload get_no_translation( new_addr_type block_addr, size_t obj_size ) const;
    void handle_new_policy( vm_page_mapping_payload* best, new_addr_type block_addr, region_entry &e, shader_page_reordering_tlb& tlb );

    // communication to L2 and TLB
    mem_fetch_interface *m_icnt;
    enum mem_fetch_status m_send_status;
    std::list<mem_fetch*> m_request_fifo;
    mem_fetch_allocator *m_mf_allocator;
    unsigned m_core_id;
    std::list<vm_page_mapping_payload*> m_tlb_update_queue;

    // hot element profiling stuff
    size_t m_l1_cache_line_size;
    struct hit_vec_and_start_of_first_page {
       std::vector< unsigned > vec;
       new_addr_type start_of_first_page;
    };
    std::map< size_t, hit_vec_and_start_of_first_page > m_obj_size_to_hit_vector;
    vm_policy_manager_stats m_stats;
};

class region_cache_config : public cache_config {
public:
    region_cache_config() { }

    virtual void init()
    {
        unsigned enabled=0;
        if ( strncmp( m_config_string, "none", 4 ) == 0 ) {
            m_valid = false;
            m_enabled = false;
            return;
        }
        sscanf(m_config_string,"%u", &enabled);
        const char *config = strchr(m_config_string,',');
        assert(config);
        cache_config::init(++config);
        m_enabled = (enabled != 0);
    }
    bool enabled() const { return m_enabled; }
private:
    bool m_enabled;
};

class region_cache {
public:
    region_cache( const region_cache_config &config ) : m_config(config), m_tags(config,-1,-1,false)
    {
        m_data = new vm_page_mapping_payload[ config.get_num_lines() ];
    }

    enum cache_request_status access( mem_fetch *mf, unsigned time, vm_page_mapping_payload &region_policy )
    {
        unsigned idx;
        enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
        if( status == HIT ) {
            region_policy = m_data[idx];
            m_tags.access(mf->get_addr(),time,idx,mf); // update LRU stack
        }
        return status;
    }
    void fill( mem_fetch *mf, unsigned time, vm_page_mapping_payload &region_policy, bool &evicted, vm_page_mapping_payload &evicted_policy )
    {
       unsigned idx;
       enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
       if( status == MISS ) {
          if( m_tags.get_block(idx).m_status == VALID ) {
              evicted = true;
              evicted_policy = m_data[idx];
          }
          m_tags.fill( mf->get_addr(), time );
       }
       m_data[idx] = region_policy;
    }
    void invalidate( mem_fetch *mf )
    {
        unsigned idx;
        enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
        assert( status == HIT );
        m_tags.get_block(idx).m_status = INVALID;
    }
private:
    const region_cache_config &m_config;
    tag_array m_tags;
    vm_page_mapping_payload *m_data;
    std::list<mem_fetch*> m_response_queue;
};

#if 0
class dynamic_tile_data_cache : public data_cache {
public:
    dynamic_tile_data_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status ) 
    : data_cache(name,config,core_id,type_id,memport,mfcreator,status)
    {
    }

    virtual enum cache_request_status access( new_addr_type m_addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) 
    {
        mem_fetch_payload *p = mf->get_payload();
        vm_page_mapping *req_mapping = dynamic_cast<vm_page_mapping*>(p);

        if( req_mapping->get_translation_config() == DYNAMIC_TILED_TRANSLATION ) {
            new_addr_type block_addr = m_config.block_addr(m_addr);
            unsigned cache_index = (unsigned)-1;
            enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
            if( status == HIT ) {
                cache_block_t &blk = m_tag_array->get_block(cache_index);
                if( blk.m_policy != *req_mapping ) 
                    status = MISS;
                else {
                    // check if specific requested stuff is in block...
                    
                    
                }
            }
            if( status == MISS ) {
                // do miss processing...
                // send request to L2...
            } else {
                // do hit processing...
            }
        }
    }
};
#endif


#endif
