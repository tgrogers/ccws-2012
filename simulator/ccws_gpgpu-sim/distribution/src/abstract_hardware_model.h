// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh,
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

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space,
   instruction_space
};

#ifdef __cplusplus

#include <string.h>
#include <stdio.h>
#include <boost/dynamic_bitset.hpp>

class mem_fetch;

// tgrogers, prob bad place for this, but tossing it in here now since everyone can see the declaration
struct sourceline_and_hit_num
{
   int sl;
   int hn;
   int instr_until_use;
   int times_through;
   int line_used;

   sourceline_and_hit_num()
      : sl( -1 ), hn( 0 ), instr_until_use( 0 ),times_through(0), line_used(-1) {}
};

typedef unsigned long long new_addr_type;
typedef unsigned address_type;
typedef unsigned addr_t;

#include <map>
struct bin_with_redundant_count
{
   std::map< new_addr_type, unsigned int > m_map;
   int redundant_reads;

   bin_with_redundant_count() : redundant_reads( 0 ) {}
};

// the following are operations the timing model can see 

enum uarch_op_t {
   NO_OP=-1,
   ALU_OP=1,
   SFU_OP,
   ALU_SFU_OP,
   LOAD_OP,
   STORE_OP,
   BRANCH_OP,
   BARRIER_OP,
   MEMORY_BARRIER_OP,
   CALL_OPS,
   RET_OPS
};
typedef enum uarch_op_t op_type;

enum _memory_op_t {
	no_memory_op = 0,
	memory_load,
	memory_store
};

#include <bitset>
#include <list>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <map>

#if !defined(__VECTOR_TYPES_H__)
struct dim3 {
   unsigned int x, y, z;
};
#endif

#if 0

// detect gcc 4.3 and use unordered map (part of c++0x)
// unordered map doesn't play nice with _GLIBCXX_DEBUG, just use a map if its enabled.
#if  defined( __GNUC__ ) and not defined( _GLIBCXX_DEBUG )
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
   #include <unordered_map>
   #define my_hash_map std::unordered_map
#else
   #include <ext/hash_map>
   namespace std {
      using namespace __gnu_cxx;
   }
   #define my_hash_map std::hash_map
#endif
#else
   #include <map>
   #define my_hash_map std::map
   #define USE_MAP
#endif

#endif

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound);

class kernel_info_t {
public:
//   kernel_info_t()
//   {
//      m_valid=false;
//      m_kernel_entry=NULL;
//      m_uid=0;
//      m_num_cores_running=0;
//      m_param_mem=NULL;
//   }
   kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry );
   ~kernel_info_t();

   void inc_running() { m_num_cores_running++; }
   void dec_running()
   {
       assert( m_num_cores_running > 0 );
       m_num_cores_running--; 
   }
   bool running() const { return m_num_cores_running>0; }
   bool done() const 
   {
       return no_more_ctas_to_run() && !running();
   }
   class function_info *entry() { return m_kernel_entry; }
   const class function_info *entry() const { return m_kernel_entry; }

   size_t num_blocks() const
   {
      return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
   }

   size_t threads_per_cta() const
   {
      return m_block_dim.x * m_block_dim.y * m_block_dim.z;
   } 

   dim3 get_grid_dim() const { return m_grid_dim; }
   dim3 get_cta_dim() const { return m_block_dim; }

   void increment_cta_id() 
   { 
      increment_x_then_y_then_z(m_next_cta,m_grid_dim); 
      m_next_tid.x=0;
      m_next_tid.y=0;
      m_next_tid.z=0;
   }
   dim3 get_next_cta_id() const { return m_next_cta; }
   bool no_more_ctas_to_run() const 
   {
      return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y || m_next_cta.z >= m_grid_dim.z );
   }

   void increment_thread_id() { increment_x_then_y_then_z(m_next_tid,m_block_dim); }
   dim3 get_next_thread_id_3d() const  { return m_next_tid; }
   unsigned get_next_thread_id() const 
   { 
      return m_next_tid.x + m_block_dim.x*m_next_tid.y + m_block_dim.x*m_block_dim.y*m_next_tid.z;
   }
   bool more_threads_in_cta() const 
   {
      return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y && m_next_tid.x < m_block_dim.x;
   }
   unsigned get_uid() const { return m_uid; }
   std::string name() const;

   std::list<class ptx_thread_info *> &active_threads() { return m_active_threads; }
   class memory_space *get_param_memory() { return m_param_mem; }

private:
   kernel_info_t( const kernel_info_t & ); // disable copy constructor
   void operator=( const kernel_info_t & ); // disable copy operator

   class function_info *m_kernel_entry;

   unsigned m_uid;
   static unsigned m_next_uid;

   dim3 m_grid_dim;
   dim3 m_block_dim;
   dim3 m_next_cta;
   dim3 m_next_tid;

   unsigned m_num_cores_running;

   std::list<class ptx_thread_info *> m_active_threads;
   class memory_space *m_param_mem;
};

struct core_config {
    core_config() 
    { 
        m_valid = false; 
        num_shmem_bank=16; 
    }
    virtual void init() = 0;

    bool m_valid;
    unsigned warp_size;

    // off-chip memory request architecture parameters
    int gpgpu_coalesce_arch;

    // shared memory bank conflict checking parameters
    static const address_type WORD_SIZE=4;
    unsigned num_shmem_bank;
    unsigned shmem_bank_func(address_type addr) const
    {
        return ((addr/WORD_SIZE) % num_shmem_bank);
    }
    unsigned mem_warp_parts;  
    unsigned gpgpu_shmem_size;

    // texture and constant cache line sizes (used to determine number of memory accesses)
    unsigned gpgpu_cache_texl1_linesize;
    unsigned gpgpu_cache_constl1_linesize;

	unsigned gpgpu_max_insn_issue_per_warp;
};

// bounded stack that implements simt reconvergence using pdom mechanism from MICRO'07 paper
const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK  MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

class simt_stack {
public:
    simt_stack( unsigned wid,  unsigned warpSize);

    void reset();
    void launch( address_type start_pc, const simt_mask_t &active_mask );
    void update( simt_mask_t &thread_done, addr_vector_t &next_pc, address_type recvg_pc );

    const simt_mask_t &get_active_mask() const;
    void     get_pdom_stack_top_info( unsigned *pc, unsigned *rpc ) const;
    unsigned get_rp() const;
    void     print(FILE*fp) const;

protected:
    unsigned m_warp_id;
    unsigned m_stack_top;
    unsigned m_warp_size;
    
    address_type *m_pc;
    simt_mask_t  *m_active_mask;
    address_type *m_recvg_pc;
    unsigned int *m_calldepth;
    
    unsigned long long  *m_branch_div_cycle;
};

#define GLOBAL_HEAP_START 0x80000000
   // start allocating from this address (lower values used for allocating globals in .ptx file)
#define SHARED_MEM_SIZE_MAX (64*1024)
#define LOCAL_MEM_SIZE_MAX (8*1024)
#define MAX_STREAMING_MULTIPROCESSORS 64
#define MAX_THREAD_PER_SM 2048
#define TOTAL_LOCAL_MEM_PER_SM (MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define TOTAL_SHARED_MEM (MAX_STREAMING_MULTIPROCESSORS*SHARED_MEM_SIZE_MAX)
#define TOTAL_LOCAL_MEM (MAX_STREAMING_MULTIPROCESSORS*MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define SHARED_GENERIC_START (GLOBAL_HEAP_START-TOTAL_SHARED_MEM)
#define LOCAL_GENERIC_START (SHARED_GENERIC_START-TOTAL_LOCAL_MEM)
#define STATIC_ALLOC_LIMIT (GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM+TOTAL_SHARED_MEM))

#if !defined(__CUDA_RUNTIME_API_H__)

enum cudaChannelFormatKind {
   cudaChannelFormatKindSigned,
   cudaChannelFormatKindUnsigned,
   cudaChannelFormatKindFloat
};

struct cudaChannelFormatDesc {
   int                        x;
   int                        y;
   int                        z;
   int                        w;
   enum cudaChannelFormatKind f;
};

struct cudaArray {
   void *devPtr;
   int devPtr32;
   struct cudaChannelFormatDesc desc;
   int width;
   int height;
   int size; //in bytes
   unsigned dimensions;
};

enum cudaTextureAddressMode {
   cudaAddressModeWrap,
   cudaAddressModeClamp
};

enum cudaTextureFilterMode {
   cudaFilterModePoint,
   cudaFilterModeLinear
};

enum cudaTextureReadMode {
   cudaReadModeElementType,
   cudaReadModeNormalizedFloat
};

struct textureReference {
   int                           normalized;
   enum cudaTextureFilterMode    filterMode;
   enum cudaTextureAddressMode   addressMode[2];
   struct cudaChannelFormatDesc  channelDesc;
};

#endif

class gpgpu_functional_sim_config 
{
public:
    void reg_options(class OptionParser * opp);

    void ptx_set_tex_cache_linesize(unsigned linesize);

    unsigned get_forced_max_capability() const { return m_ptx_force_max_capability; }
    bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
    bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }

    int         get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
    const char* get_ptx_inst_debug_file() const  { return g_ptx_inst_debug_file; }
    int         get_ptx_inst_debug_thread_uid() const { return g_ptx_inst_debug_thread_uid; }
    unsigned    get_texcache_linesize() const { return m_texcache_linesize; }

private:
    // PTX options
    int m_ptx_convert_to_ptxplus;
    int m_ptx_use_cuobjdump;
    unsigned m_ptx_force_max_capability;

    int   g_ptx_inst_debug_to_file;
    char* g_ptx_inst_debug_file;
    int   g_ptx_inst_debug_thread_uid;

    unsigned m_texcache_linesize;
};

// Defines if we are using virtual translation based on programmer input (STATIC_TRANSLATION), hardware detection (DYNAMIC_TRANSLATION) or not at all (NO_TRANSLATION)
enum translation_config {
      NO_TRANSLATION = 0,
      STATIC_TRANSLATION,
      DYNAMIC_TRANSLATION
};

// Defines how we have translated the pages.
enum virtual_page_type {
    VP_NON_TRANSLATED = 0, // the linear address is just the virtual address.
    VP_TILED, // swizzled all the data in the object into a tiled format (90 degree malloc)
    VP_HOT_DATA_ALIGNED, // group the hot bytes from each object together and leave all the cold data at the end.  Objects must be page aligned
    VP_HOT_DATA_UNALIGNED, // same as above but the objects can be unaligned, space on the start and end of the pages is left cold.
    VP_TYPE_MAX
};

class mem_fetch_payload {
public:
    virtual size_t size() = 0;
    virtual ~mem_fetch_payload() {}
};

class base_virtual_page {
public:
    virtual ~base_virtual_page() {}

    // Translate virtual address to linear address
    virtual new_addr_type translate( const new_addr_type virtual_addr ) const = 0;

    virtual bool operator==( const base_virtual_page &another ) const
    {
        if( m_start != another.m_start ) return false;
        if( m_end != another.m_end ) return false;
        return true;
    }

    virtual bool operator!=( const base_virtual_page &another ) const {
        return !(*this == another);
    }

    bool contains( new_addr_type addr ) const {
        return addr >= (new_addr_type)m_start && addr < (new_addr_type)m_end;
    }

    virtual void print(FILE *fp) const
    {
        fprintf( fp,"base_virtual_page:\nstart=0x%p\nend=0x%p\n", m_start, m_end );
    }

    void* get_start_addr() const { return m_start; }
    void* get_end_addr() const { return m_end; }

protected:
    base_virtual_page()
        : m_start( NULL ), m_end( NULL ) {}
    base_virtual_page( void *s, void *e )
        : m_start( s ), m_end( e ) {}

    void* m_start;
    void* m_end;
};

class virtual_object_divided_page : public base_virtual_page {
public:
    virtual ~virtual_object_divided_page(){}

    virtual void print( FILE *fp ) const
    {
        base_virtual_page::print( fp );
        fprintf( fp,"m_object_size=%zu\n", m_object_size );
    }

    virtual bool operator==( const base_virtual_page &another ) const {
        if ( !base_virtual_page::operator ==( another ) ) return false;
        const virtual_object_divided_page* p_another = dynamic_cast< const virtual_object_divided_page* >( &another );
        if ( !p_another ) return false;
        if ( m_object_size != p_another->m_object_size ) return false;
        return true;
    }

    size_t get_object_size() const { return m_object_size; }

protected:
    virtual_object_divided_page( void *s, void *e, size_t os )
            : base_virtual_page( s, e ), m_object_size( os ) {
        assert( os );
    }

    size_t m_object_size;
};

class virtual_hot_page : public virtual_object_divided_page {
public:
    virtual ~virtual_hot_page() {}

    virtual void print( FILE *fp ) const
    {
        virtual_object_divided_page::print( fp );
        fprintf(fp,"m_hotbytes.count()=%zu\n", m_hotbytes.count() );
        unsigned zero_count = 0;
        for ( unsigned i = 0; i < m_hotbytes.size(); ++i ) {
            if ( m_hotbytes.test(i) ) {
                if ( zero_count > 0 )
                    printf("%u zeros, ", zero_count);
                    printf("1, ");
                    zero_count = 0;
                } else {
                    zero_count++;
                }
        }
        if ( zero_count > 0 )
        printf("%u zeros", zero_count);
        printf("\n");
    }

    bool operator==( const base_virtual_page &another ) const
    {
        if ( !virtual_object_divided_page::operator ==( another ) ) return false;
        const virtual_hot_page* p_another = dynamic_cast< const virtual_hot_page* >( &another );
        if ( !p_another ) return false;
        if ( m_hotbytes != p_another->m_hotbytes ) return false;
        if ( m_page_size != p_another->m_page_size ) return false;
        return true;
    }

    virtual new_addr_type reverse_translate( new_addr_type unswizzled ) const = 0;

    size_t get_hot_data_size() const {
        return m_hotbytes.count();
    }

protected:
    virtual_hot_page( void *s, void *e, size_t os, const boost::dynamic_bitset<>& hb, size_t ps )
            : virtual_object_divided_page( s, e, os ), m_hotbytes( hb ), m_page_size( ps ) {
        assert( m_page_size && hb.count() > 0 );
    }

    boost::dynamic_bitset<> m_hotbytes;
    size_t m_page_size;
};

class virtual_page : public base_virtual_page {
public:
    virtual_page( void *s, void *e ) : base_virtual_page( s, e ) {
        assert( ( size_t )e - ( size_t )s > 0 );
    }
    virtual ~virtual_page(){}

    virtual new_addr_type translate( const new_addr_type virtual_addr ) const {
        return virtual_addr;
    }
};

struct virtual_access_tile_identifier {
    virtual_access_tile_identifier()
      : tile_num(-1), struct_num(-1), offset(-1) {}
   int tile_num;
   int struct_num;
   int offset;
};

class virtual_tiled_page : public virtual_object_divided_page {
public:
    virtual_tiled_page( void *s, void *e, size_t os, size_t fs, size_t tw )
        : virtual_object_divided_page( s, e, os ), m_field_size( fs ), m_tile_width( tw ) {
        assert( fs && tw );
    }
    virtual ~virtual_tiled_page() {}

    virtual void print( FILE *fp ) const
    {
        virtual_object_divided_page::print( fp );
        fprintf( fp,"m_field_size=%zu\nm_tile_width=%zu\n", m_field_size, m_tile_width );
    }

    bool operator==( const base_virtual_page &another ) const
    {
        if ( !virtual_object_divided_page::operator ==( another ) ) return false;
        const virtual_tiled_page* p_another = dynamic_cast< const virtual_tiled_page* >( &another );
        if ( !p_another ) return false;
        if ( m_field_size != p_another->m_field_size ) return false;
        if ( m_tile_width != p_another->m_tile_width ) return false;
        return true;
    }

    virtual new_addr_type translate( const new_addr_type virtual_addr ) const;

    size_t get_tile_width() const { return m_tile_width; }
    size_t get_field_size() const { return m_field_size; }

protected:

    new_addr_type translate(const virtual_access_tile_identifier &ident) const;
    virtual_access_tile_identifier translate_to_ident( new_addr_type virtual_addr ) const;

    size_t m_field_size;
    size_t m_tile_width;
};

class virtual_aligned_hot_page : public virtual_hot_page {
public:
    virtual_aligned_hot_page( void *s, void *e, size_t os, const boost::dynamic_bitset<>& hb, size_t ps )
            : virtual_hot_page( s, e, os, hb, ps ) {}
    virtual ~virtual_aligned_hot_page() {}

    virtual new_addr_type translate( const new_addr_type virtual_addr ) const;

    virtual new_addr_type reverse_translate( new_addr_type unswizzled ) const {
        return unswizzled;
    }
};

class virtual_unaligned_hot_page : public virtual_hot_page {
public:
    virtual_unaligned_hot_page( void *s, void *e, size_t os, const boost::dynamic_bitset<>& hb, size_t ps, size_t cbs, new_addr_type start_first_page )
            : virtual_hot_page( s, e, os, hb, ps ), m_cache_block_size( cbs ), m_start_of_first_page( start_first_page ) {
        assert( cbs && start_first_page );
    }
    virtual ~virtual_unaligned_hot_page() {}

    virtual new_addr_type translate( const new_addr_type virtual_addr ) const;

    bool operator==( const base_virtual_page &another ) const
    {
        if ( !virtual_hot_page::operator ==( another ) ) return false;
        const virtual_unaligned_hot_page* p_another = dynamic_cast< const virtual_unaligned_hot_page* >( &another );
        if ( !p_another ) return false;
        if ( m_cache_block_size != p_another->m_cache_block_size ) return false;
        if ( m_start_of_first_page != p_another->m_start_of_first_page ) return false;
        return true;
    }

    virtual new_addr_type reverse_translate( new_addr_type unswizzled ) const;

private:
    static const size_t FORMAT_DATA_SIZE = 0;

    size_t m_cache_block_size;
    new_addr_type m_start_of_first_page;
};


struct virtual_page_creation_t {
    virtual_page_creation_t()
        : m_type( VP_NON_TRANSLATED ),
          m_start( 0 ),
          m_end( 0 ),
          m_object_size( 0 ),
          m_page_size( 0 ),
          m_field_size( 0 ),
          m_tile_width( 0 ),
          m_cache_block_size( 0 ),
          m_start_of_first_page( 0 ) {}

    virtual_page_type m_type;
    // For Abstract classes
    //-------------------------------------------
    // for base_virtual_page
    void* m_start;
    void* m_end;

    // for virtual_object_divided_page
    size_t m_object_size;

    // for virtual_hot_page
    boost::dynamic_bitset<> m_hotbytes;
    size_t m_page_size;
    //-------------------------------------------

    // For Concrete classes
    //-------------------------------------------
    // for virtual_tiled_page
    size_t m_field_size;
    size_t m_tile_width;

    // for virtual_unaligned_hot_page
    size_t m_cache_block_size;
    new_addr_type m_start_of_first_page;
    //-------------------------------------------
};

class virtual_page_factory {
public:
    virtual_page_factory() : m_created_pages( 0 ), m_deleted_pages( 0 ) {
        for ( unsigned i = 0; i < VP_TYPE_MAX; ++i ) {
            m_per_type_pages_created[ i ] = 0;
        }
    }

    // Right now we only support one GPU and one page factory
    static virtual_page_factory* get_global_factory();

    base_virtual_page* create_new_page( const virtual_page_creation_t& new_page );
    base_virtual_page* copy_page( const base_virtual_page* page );

    void destroy_page( base_virtual_page* page ) {
        if ( page ) {
            delete page;
            m_deleted_pages++;
        }
    }

    void print_creation_page_stats( FILE* fp ) const {
        fprintf( fp, "virtual_page_factory::m_created_pages = %u\n", m_created_pages );
        for ( unsigned i = 0; i < VP_TYPE_MAX; ++i ) {
            fprintf( fp, "virtual_page_factory::m_per_type_pages_created[ %d ] = %d\n", i, m_per_type_pages_created[ i ] );
        }
        fprintf( fp, "virtual_page_factory::m_deleted_pages = %u\n", m_deleted_pages );
        fprintf( fp, "virtual_page_factory::outstanding_pages = %u\n", m_created_pages - m_deleted_pages );

    }

private:
    unsigned m_created_pages;
    unsigned m_deleted_pages;
    unsigned m_per_type_pages_created[ VP_TYPE_MAX ];
};

// The result of passed by the predictor
class vm_page_mapping_payload : public mem_fetch_payload {
public:
    vm_page_mapping_payload();
    vm_page_mapping_payload( virtual_page_type config, const virtual_page_creation_t& new_page );
    vm_page_mapping_payload( virtual_page_type config, const base_virtual_page* page );
    vm_page_mapping_payload( const vm_page_mapping_payload& to_copy );
    virtual ~vm_page_mapping_payload();

    base_virtual_page* get_virtualized_page() const { return m_virtualized_page; }
    virtual_page_type get_page_type() const { return m_vm_page_type; }

    bool operator==( const vm_page_mapping_payload& rhs ) const {
        if ( m_vm_page_type != rhs.m_vm_page_type ) return false;
        if ( m_virtualized_page == NULL && rhs.m_virtualized_page !=NULL ) return false;
        if ( rhs.m_virtualized_page == NULL && m_virtualized_page !=NULL ) return false;
        if ( m_virtualized_page && rhs.m_virtualized_page ) {
            if ( *m_virtualized_page != *rhs.m_virtualized_page ) return false;
        }
        return true;
    }

    vm_page_mapping_payload& operator=( const vm_page_mapping_payload& rhs );

    virtual size_t size()
    {
        // number of bytes if this mapping is sent as a packet on interconnect
        return 4; // TODO: adjust
    }

    bool operator!=( const vm_page_mapping_payload& rhs ) const {
        return !(*this == rhs);
    }

    void clear() {
        m_vm_page_type = VP_NON_TRANSLATED;
        m_virtualized_page = NULL;
    }

    void print(FILE* fp) const {
        fprintf(fp,"m_vm_page_type=%d\n", m_vm_page_type);
        m_virtualized_page->print(fp);
    }

private:
    virtual_page_type m_vm_page_type;
    base_virtual_page* m_virtualized_page;
};

class vm_exploded_L2_payload : public mem_fetch_payload {
public:
   vm_exploded_L2_payload( mem_fetch* new_fetch ) : m_original_fetch( new_fetch ), m_num_exploded_in_flight( 0 ) {}
   virtual ~vm_exploded_L2_payload(){}

   virtual size_t size()
   {
       // number of bytes if this mapping is sent as a packet on interconnect
       return 4; // TODO: adjust
   }

   mem_fetch* get_original_fetch() { return m_original_fetch; }
   void increment_exploded_in_flight() { ++m_num_exploded_in_flight; }
   void decrement_exploded_in_flight() { --m_num_exploded_in_flight; }
   unsigned get_exploded_in_flight() { return m_num_exploded_in_flight; }
private:
   mem_fetch* m_original_fetch;
   unsigned m_num_exploded_in_flight;
};

extern translation_config g_translation_config;
extern virtual_page_type g_translated_page_type;

class gpgpu_t {
public:
    gpgpu_t( const gpgpu_functional_sim_config &config );
    void* gpu_malloc( size_t size );
    void* gpu_mallocarray( size_t count );
    void  gpu_free( void *devPtr );
    void  gpu_memset( size_t dst_start_addr, int c, size_t count );
    void  memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
    void  memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
    void  memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
    
    class memory_space *get_global_memory() { return m_global_mem; }
    class memory_space *get_tex_memory() { return m_tex_mem; }
    class memory_space *get_surf_memory() { return m_surf_mem; }

    void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
    void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
    const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);

    const struct textureReference* get_texref(const std::string &texname) const
    {
        std::map<std::string, const struct textureReference*>::const_iterator t=m_NameToTextureRef.find(texname);
        assert( t != m_NameToTextureRef.end() );
        return t->second;
    }
    const struct cudaArray* get_texarray( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*,const struct cudaArray*>::const_iterator t=m_TextureRefToCudaArray.find(texref);
        assert(t != m_TextureRefToCudaArray.end());
        return t->second;
    }
    const struct textureInfo* get_texinfo( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*, const struct textureInfo*>::const_iterator t=m_TextureRefToTexureInfo.find(texref);
        assert(t != m_TextureRefToTexureInfo.end());
        return t->second;
    }

    const gpgpu_functional_sim_config &get_config() const { return m_function_model_config; }
    FILE* get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

    virtual_object_divided_page* get_virtualized_page( new_addr_type addr );

    void set_static_virtualized_page( virtual_object_divided_page* page );
    void set_dynamic_virtualized_page( virtual_object_divided_page* page );
    void remove_virtualized_page( new_addr_type addr );

    virtual_page_factory& get_page_factory() { return m_page_factory; }

    void print_virtualized_pages( FILE* fp, bool print_all_pages );

    void handle_static_virtual_page( void* start_of_vm,
                    size_t total_size,
                    size_t field_size,
                    size_t tile_width,
                    size_t object_size,
                    size_t page_size,
                    boost::dynamic_bitset<> hotbytes );

protected:
    void set_page( virtual_object_divided_page* page );

    const gpgpu_functional_sim_config &m_function_model_config;
    FILE* ptx_inst_debug_file;

    class memory_space *m_global_mem;
    class memory_space *m_tex_mem;
    class memory_space *m_surf_mem;
    
    unsigned long long m_dev_malloc;
    
    std::map<std::string, const struct textureReference*> m_NameToTextureRef;
    std::map<const struct textureReference*,const struct cudaArray*> m_TextureRefToCudaArray;
    std::map<const struct textureReference*, const struct textureInfo*> m_TextureRefToTexureInfo;

    virtual_page_factory m_page_factory;
    std::map< new_addr_type, virtual_object_divided_page* > m_virtualized_mem_pages;
};

struct gpgpu_ptx_sim_kernel_info 
{
   // Holds properties of the kernel (Kernel's resource use). 
   // These will be set to zero if a ptxinfo file is not present.
   int lmem;
   int smem;
   int cmem;
   int regs;
   unsigned ptx_version;
   unsigned sm_target;
};

struct gpgpu_ptx_sim_arg {
   gpgpu_ptx_sim_arg() { m_start=NULL; }
   gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset)
   {
      m_start=arg;
      m_nbytes=size;
      m_offset=offset;
   }
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

class memory_space_t {
public:
   memory_space_t() { m_type = undefined_space; m_bank=0; }
   memory_space_t( const enum _memory_space_t &from ) { m_type = from; m_bank = 0; }
   bool operator==( const memory_space_t &x ) const { return (m_bank == x.m_bank) && (m_type == x.m_type); }
   bool operator!=( const memory_space_t &x ) const { return !(*this == x); }
   bool operator<( const memory_space_t &x ) const 
   { 
      if(m_type < x.m_type)
         return true;
      else if(m_type > x.m_type)
         return false;
      else if( m_bank < x.m_bank )
         return true; 
      return false;
   }
   enum _memory_space_t get_type() const { return m_type; }
   unsigned get_bank() const { return m_bank; }
   void set_bank( unsigned b ) { m_bank = b; }
   bool is_const() const { return (m_type == const_space) || (m_type == param_space_kernel); }
   bool is_local() const { return (m_type == local_space) || (m_type == param_space_local); }

private:
   enum _memory_space_t m_type;
   unsigned m_bank; // n in ".const[n]"; note .const == .const[0] (see PTX 2.1 manual, sec. 5.1.3)
};

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

enum mem_access_type {
   GLOBAL_ACC_R, 
   LOCAL_ACC_R, 
   CONST_ACC_R, 
   TEXTURE_ACC_R, 
   GLOBAL_ACC_W, 
   LOCAL_ACC_W,
   L1_WRBK_ACC,
   L2_WRBK_ACC, 
   INST_ACC_R,
   VM_POLICY_CHANGE_REQ,
   VM_POLICY_CHANGE_REQ_RC,
   VM_POLICY_CHANGE_ACK_RC,
   NUM_MEM_ACCESS_TYPE
};

enum cache_operator_type {
    CACHE_UNDEFINED, 

    // loads
    CACHE_ALL,          // .ca
    CACHE_LAST_USE,     // .lu
    CACHE_VOLATILE,     // .cv
                       
    // loads and stores 
    CACHE_STREAMING,    // .cs
    CACHE_GLOBAL,       // .cg

    // stores
    CACHE_WRITE_BACK,   // .wb
    CACHE_WRITE_THROUGH // .wt
};

#define NUM_SECTORED 8
class mem_access_t {
public:
   mem_access_t() { init(); }
   mem_access_t( mem_access_type type, 
                 address_type address, 
                 unsigned size,
                 bool wr )
   {
       init();
       m_type = type;
       m_addr = address;
       m_req_size = size;
       m_write = wr;
   }
   mem_access_t( mem_access_type type, 
                 address_type address, 
                 unsigned size, 
                 bool wr, 
                 const active_mask_t &active_mask,
                 const mem_access_byte_mask_t &byte_mask )
    : m_warp_mask(active_mask), m_byte_mask(byte_mask)
   {
      init();
      m_type = type;
      m_addr = address;
      m_req_size = size;
      m_write = wr;
   }

   new_addr_type get_addr() const { return m_addr; }
   unsigned get_size() const { return m_req_size; }
   const active_mask_t &get_warp_mask() const { return m_warp_mask; }
   bool is_write() const { return m_write; }
   enum mem_access_type get_type() const { return m_type; }
   const mem_access_byte_mask_t &get_byte_mask() const { return m_byte_mask; }

   void print(FILE *fp) const
   {
       fprintf(fp,"addr=0x%llx, %s, size=%u, ", m_addr, m_write?"store":"load ", m_req_size );
       switch(m_type) {
       case GLOBAL_ACC_R:  fprintf(fp,"GLOBAL_R"); break;
       case LOCAL_ACC_R:   fprintf(fp,"LOCAL_R "); break;
       case CONST_ACC_R:   fprintf(fp,"CONST   "); break;
       case TEXTURE_ACC_R: fprintf(fp,"TEXTURE "); break;
       case GLOBAL_ACC_W:  fprintf(fp,"GLOBAL_W"); break;
       case LOCAL_ACC_W:   fprintf(fp,"LOCAL_W "); break;
       case L2_WRBK_ACC:   fprintf(fp,"L2_WRBK "); break;
       case INST_ACC_R:    fprintf(fp,"INST    "); break;
       case L1_WRBK_ACC:   fprintf(fp,"L1_WRBK "); break;
       default:            fprintf(fp,"unknown "); break;
       }
   }

   void set_diverged() { m_diverged=true; }
   bool is_diverged() const { return m_diverged; }
   void set_is_ptr() { m_is_prt = true; }
   bool is_ptr() const { return m_is_prt; }
   void set_vm_page( vm_page_mapping_payload* new_page ) { m_virtual_page = new_page; }
   vm_page_mapping_payload* get_vm_page() const { return m_virtual_page; }

   // tayler: Memory Divergence info
   void set_byte_mask_start(unsigned _start){byte_mask_start = _start;}
   unsigned get_byte_mask_start() const {return byte_mask_start; }
   void set_byte_mask_flag(){byte_mask_flag = true;}
   bool is_byte_mask_flag() const {return byte_mask_flag;}


	// Sectored cache
	void set_sector(unsigned index) {m_sector_mask.set(index);}
	void set_sector(std::bitset<NUM_SECTORED> t){m_sector_mask = t;}
	bool is_sector_set(unsigned index) const{ return m_sector_mask.test(index);}
	std::bitset<NUM_SECTORED> get_sector() const {return m_sector_mask;}

private:
   void init() 
   {
      m_uid=++sm_next_access_uid;
      m_addr=0;
      m_req_size=0;
      m_diverged=false;
      m_is_prt = false;
      byte_mask_start = 0;
      byte_mask_flag = false;
      m_virtual_page = NULL;
	  m_sector_mask.reset();
   }

   unsigned      m_uid;
   new_addr_type m_addr;     // request address
   bool          m_write;
   bool          m_is_prt;
   unsigned      m_req_size; // bytes
   mem_access_type m_type;
   active_mask_t m_warp_mask;
   mem_access_byte_mask_t m_byte_mask;
   bool m_diverged;

   static unsigned sm_next_access_uid;

   vm_page_mapping_payload* m_virtual_page;

   // tayler: Memory Divergence Metric data
   unsigned		byte_mask_start;
   bool		byte_mask_flag;

	// Sectored cache info
	std::bitset<NUM_SECTORED> m_sector_mask; // (128 / 8 = 16 byte sectors)
};

class mem_fetch;
class mem_fetch_payload;

class mem_fetch_interface {
public:
    virtual bool full( unsigned size, bool write, mem_fetch* fetch = NULL ) const = 0;
    virtual void push( mem_fetch *mf ) = 0;
};

class mem_fetch_allocator {
public:
    virtual mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr, mem_fetch_payload *payload=NULL ) const = 0;
    virtual mem_fetch *alloc( const class warp_inst_t &inst, const mem_access_t &access, mem_fetch_payload *payload=NULL ) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a instruction
#define MAX_REG_OPERANDS 8

struct dram_callback_t {
   dram_callback_t() { function=NULL; instruction=NULL; thread=NULL; }
   void (*function)(const class inst_t*, class ptx_thread_info*);
   const class inst_t* instruction;
   class ptx_thread_info *thread;
};

class inst_t {
public:
    inst_t()
    {
        m_decoded=false;
        pc=(address_type)-1;
        reconvergence_pc=(address_type)-1;
        op=NO_OP; 
        memset(out, 0, sizeof(unsigned)); 
        memset(in, 0, sizeof(unsigned)); 
        is_vectorin=0; 
        is_vectorout=0;
        space = memory_space_t();
        cache_op = CACHE_UNDEFINED;
        latency = 1;
        initiation_interval = 1;
        for( unsigned i=0; i < MAX_REG_OPERANDS; i++ ) {
            arch_reg.src[i] = -1;
            arch_reg.dst[i] = -1;
        }
        isize=0;
        disp=0;
    }
    bool valid() const { return m_decoded; }
    virtual void print_insn( FILE *fp ) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
    }
    bool is_load() const { return (op == LOAD_OP || memory_op == memory_load); }
    bool is_store() const { return (op == STORE_OP || memory_op == memory_store); }
    unsigned get_memory_displacement() const
    { 
        assert(op==LOAD_OP||op==STORE_OP); 
        return disp; 
    }

    address_type pc;        // program counter address of instruction
    unsigned isize;         // size of instruction in bytes 
    op_type op;             // opcode (uarch visible)
    _memory_op_t memory_op; // memory_op used by ptxplus 

    address_type reconvergence_pc; // -1 => not a branch, -2 => use function return address
    
    unsigned out[4];
    unsigned in[4];
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred; // predicate register number
    int ar1, ar2;
    // register number for bank conflict evaluation
    struct {
        int dst[MAX_REG_OPERANDS];
        int src[MAX_REG_OPERANDS];
    } arch_reg;
    //int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation
    unsigned latency; // operation latency 
    unsigned initiation_interval;

    unsigned data_size; // what is the size of the word being operated on?
    memory_space_t space;
    cache_operator_type cache_op;

    unsigned disp; // displacement for load/store

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

enum divergence_support_t {
   POST_DOMINATOR = 1,
   NUM_SIMD_MODEL
};

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

class warp_inst_t: public inst_t {
public:
    // constructors
    warp_inst_t() 
    {
        m_uid=0;
        m_empty=true; 
        m_config=NULL; 
    }
    warp_inst_t( const core_config *config ) 
    { 
        m_uid=0;
        assert(config->warp_size<=MAX_WARP_SIZE); 
        m_config=config;
        m_empty=true; 
        m_isatomic=false;
        m_per_scalar_thread_valid=false;
        m_mem_accesses_created=false;
        m_cache_hit=false;
		m_is_printf = false;
    }

    // modifiers
    void do_atomic(bool forceDo=false);
    void do_atomic( const active_mask_t& access_mask, bool forceDo=false );
    void clear() 
    { 
        m_empty=true; 
        std::vector<per_thread_info>::iterator t;
        for( t=m_per_scalar_thread.begin(); t != m_per_scalar_thread.end(); ++t ) {
           t->virtual_page_policy.clear();
        }
    }
    void issue( const active_mask_t &mask, unsigned warp_id, unsigned long long cycle ) 
    {
        m_warp_active_mask=mask;
        m_uid = ++sm_next_uid;
        m_warp_id = warp_id;
        issue_cycle = cycle;
        cycles = initiation_interval;
        m_cache_hit=false;
        m_empty=false;
    }
    void completed( unsigned long long cycle ) const;  // stat collection: called when the instruction is completed  
    void set_addr( unsigned n, new_addr_type addr ) 
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        m_per_scalar_thread[n].memreqaddr[0] = addr;
    }
    void set_addr( unsigned n, new_addr_type* addr, unsigned num_addrs )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
        for(unsigned i=0; i<num_addrs; i++)
            m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }

    struct transaction_info {
        std::bitset<4> chunks; // bitmask: 32-byte chunks accessed
        mem_access_byte_mask_t bytes;
        active_mask_t active; // threads in this transaction

        bool test_bytes(unsigned start_bit, unsigned end_bit) {
           for( unsigned i=start_bit; i<=end_bit; i++ )
              if(bytes.test(i))
                 return true;
           return false;
        }
    };

    void set_vm_policy( unsigned n, const vm_page_mapping_payload& new_policy )
    {
       if( !m_per_scalar_thread_valid ) {
          m_per_scalar_thread.resize(m_config->warp_size);
          m_per_scalar_thread_valid=true;
       }
       m_per_scalar_thread[n].virtual_page_policy = new_policy;
    }
    void generate_mem_accesses();
    void memory_coalescing_arch_13( bool is_write, mem_access_type access_type );
    void memory_coalescing_arch_13_atomic( bool is_write, mem_access_type access_type );
    void memory_coalescing_arch_13_reduce_and_send( bool is_write, mem_access_type access_type, const transaction_info &info, new_addr_type addr, unsigned segment_size );

    void add_callback( unsigned lane_id, 
                       void (*function)(const class inst_t*, class ptx_thread_info*),
                       const inst_t *inst, 
                       class ptx_thread_info *thread )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
            m_isatomic=true;
        }
        m_per_scalar_thread[lane_id].callback.function = function;
        m_per_scalar_thread[lane_id].callback.instruction = inst;
        m_per_scalar_thread[lane_id].callback.thread = thread;
    }
    void set_active( const active_mask_t &active );

    void clear_active( const active_mask_t &inactive );
    void set_not_active( unsigned lane_id );

    // accessors
    virtual void print_insn(FILE *fp) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
        for (int i=(int)m_config->warp_size-1; i>=0; i--)
            fprintf(fp, "%c", ((m_warp_active_mask[i])?'1':'0') );
    }
    bool active( unsigned thread ) const { return m_warp_active_mask.test(thread); }
    unsigned active_count() const { return m_warp_active_mask.count(); }
    bool empty() const { return m_empty; }
    unsigned warp_id() const 
    { 
        assert( !m_empty );
        return m_warp_id; 
    }
    bool has_callback( unsigned n ) const
    {
        return m_warp_active_mask[n] && m_per_scalar_thread_valid && 
            (m_per_scalar_thread[n].callback.function!=NULL);
    }
    new_addr_type get_addr( unsigned n ) const
    {
        assert( m_per_scalar_thread_valid );
        return m_per_scalar_thread[n].memreqaddr[0];
    }
    vm_page_mapping_payload get_vitrual_page_payload( unsigned n ) const
     {
         assert( m_per_scalar_thread_valid );
         return m_per_scalar_thread[n].virtual_page_policy;
     }


    void print_per_thread_info() ;

    bool isatomic() const { return m_isatomic; }

    unsigned warp_size() const { return m_config->warp_size; }

    bool accessq_empty() const { return m_accessq.empty(); }
    unsigned accessq_count() const { return m_accessq.size(); }
    const mem_access_t &accessq_back() { return m_accessq.back(); }
    void accessq_pop_back() { m_accessq.pop_back(); }

    bool dispatch_delay()
    { 
        if( cycles > 0 ) 
            cycles--;
        return cycles > 0;
    }

    void print( FILE *fout ) const;
    unsigned get_uid() const { return m_uid; }

    unsigned long long get_issue_cycle() const { return issue_cycle; }

protected:

    unsigned m_uid;
    bool m_empty;
    bool m_cache_hit;
    unsigned long long issue_cycle;
    unsigned cycles; // used for implementing initiation interval delay
    bool m_isatomic;
    bool m_is_printf;
    unsigned m_warp_id;
    const core_config *m_config; 
    active_mask_t m_warp_active_mask;

    struct per_thread_info {
        per_thread_info();
        dram_callback_t callback;
        new_addr_type memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD]; // effective address, upto 8 different requests (to support 32B access in 8 chunks of 4B each)
		vm_page_mapping_payload virtual_page_policy;
    };
    bool m_per_scalar_thread_valid;
    std::vector<per_thread_info> m_per_scalar_thread;
    bool m_mem_accesses_created;
    std::list<mem_access_t> m_accessq;

    static unsigned sm_next_uid;
};

void move_warp( warp_inst_t *&dst, warp_inst_t *&src );

size_t get_kernel_code_size( class function_info *entry );

/*
 * This abstract class used as a base for functional and performance and simulation, it has basic functional simulation
 * data structures and procedures. 
 */
class core_t {
    public:
        virtual ~core_t() {}
        virtual void warp_exit( unsigned warp_id ) = 0;
        virtual bool warp_waiting_at_barrier( unsigned warp_id ) const = 0;
        virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)=0;
        class gpgpu_sim * get_gpu() {return m_gpu;}
        void execute_warp_inst_t(warp_inst_t &inst, unsigned warpSize, unsigned warpId =(unsigned)-1);
        bool  ptx_thread_done( unsigned hw_thread_id ) const ;
        void updateSIMTStack(unsigned warpId, unsigned warpSize, warp_inst_t * inst);
        void initilizeSIMTStack(unsigned warps, unsigned warpsSize);
        warp_inst_t getExecuteWarp(unsigned warpId);

    protected:
        class gpgpu_sim *m_gpu;
        kernel_info_t *m_kernel;
        simt_stack  **m_simt_stack; // pdom based reconvergence context for each warp
        class ptx_thread_info ** m_thread; 
};
extern bool hack_use_warp_prot_list;


//register that can hold multiple instructions.
class register_set {
public:
	register_set(unsigned num, const char* name){
		for( unsigned i = 0; i < num; i++ ) {
			regs.push_back(new warp_inst_t());
		}
		m_name = name;
	}
	bool has_free(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( regs[i]->empty() ) {
				return true;
			}
		}
		return false;
	}
	bool has_ready(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( not regs[i]->empty() ) {
				return true;
			}
		}
		return false;
	}

	void move_in( warp_inst_t *&src ){
		warp_inst_t** free = get_free();
		move_warp(*free, src);
	}
	//void copy_in( warp_inst_t* src ){
		//   src->copy_contents_to(*get_free());
		//}
	void move_out_to( warp_inst_t *&dest ){
		warp_inst_t **ready=get_ready();
		move_warp(dest, *ready);
	}

	warp_inst_t** get_ready(){
		warp_inst_t** ready;
		ready = NULL;
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( not regs[i]->empty() ) {
				if( ready and (*ready)->get_uid() < regs[i]->get_uid() ) {
					// ready is oldest
				} else {
					ready = &regs[i];
				}
			}
		}
		return ready;
	}

	void print(FILE* fp) const{
		fprintf(fp, "%s : @%p\n", m_name, this);
		for( unsigned i = 0; i < regs.size(); i++ ) {
			fprintf(fp, "     ");
			regs[i]->print(fp);
			fprintf(fp, "\n");
		}
	}

	warp_inst_t ** get_free(){
		for( unsigned i = 0; i < regs.size(); i++ ) {
			if( regs[i]->empty() ) {
				return &regs[i];
			}
		}
		assert(0 && "No free registers found");
		return NULL;
	}

private:
	std::vector<warp_inst_t*> regs;
	const char* m_name;
};

#endif // #ifdef __cplusplus

#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
