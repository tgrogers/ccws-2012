// (c) tgrogers UBC December - 2011
//
//

#include "gpgpu-sim/gpu-cache.h"
#include <list>

class cache_oracle {
public:
    static const unsigned long long NEVER_REREFERENCED = 0xFFFFFFFFFFFFFFFF;
    cache_oracle( const cache_config& config,
                  const std::list< access_stream_entry >& stream )
        : m_configuration( config ), m_access_stream( stream ) {}

    unsigned long long get_future_access_position ( new_addr_type block_addr,
            std::list< access_stream_entry >::const_iterator current_pos ) const {
        unsigned long long position_from_here = 0;
        if ( current_pos != m_access_stream.end() ) {
            ++current_pos;
        }
        for(;current_pos != m_access_stream.end(); ++current_pos ) {
            if ( block_addr == m_configuration.block_addr( current_pos->get_addr() ) ) {
                return position_from_here;
            }
            ++position_from_here;
        }
        return NEVER_REREFERENCED;
    }

private:
    const cache_config& m_configuration;
    const std::list< access_stream_entry >& m_access_stream;
};
