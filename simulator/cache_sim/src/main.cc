// (c) tgrogers UBC December - 2011
//
//

#include "stdio.h"
#include <list>
#include "gpgpu-sim/gpu-cache.h"
#include "gpgpu-sim/option_parser.h"
#include <boost/algorithm/string.hpp>
#include "cache_oracle.h"

static const char CACHE_CONFIG_FILE[] = "cache_sim.config";
cache_oracle* g_cache_oracle;
std::list< access_stream_entry >::const_iterator g_access_stream_iterator;
std::list< std::map< unsigned, std::list< std::list< access_stream_entry > > > >
        g_warp_to_pc_bundle_to_access_list;

enum access_rescheduler_type
{
    ACCESS_RESCHEDULER_NONE = 0,
    ACCESS_RESCHEDULER_WARP_WITH_MOST_IMMEDIATE_HITS,// 1
    ACCESS_RESCHEDULER_WARP_WITH_LEAST_IMMEDIATE_MISSES,// 2
    ACCESS_RESCHEDULER_WARP_WITH_HIGHEST_IMMEDATE_HIT_RATE,// 3
    ACCESS_RESCHEDULER_WARP_WHERE_MOST_LANES_HIT,// 4
    ACCESS_RESCHEDULER_CONCURRENCY_LIMITED,// 5
    ACCESS_RESCHEDULER_MAX
};

struct simlator_input
{
    cache_config configuration;
    std::list< access_stream_entry > access_stream;
    unsigned enable_flushes;
    access_rescheduler_type access_rescheduler;
    unsigned do_not_cache_dead_blocks;
    unsigned do_not_evict_global_writes;
};

struct access_stats {
    access_stats() { memset( this, 0x0, sizeof( *this ) ); }
    unsigned accesses_i_initially_loaded;
    unsigned accesses_someone_else_initially_loaded;
    unsigned access_nobody_touched_yet;
    unsigned writes_to_block_previously_loaded;
    unsigned writes_to_previouly_unreferrenced_blocks;
    unsigned redundant_this_pc_loaded;
    unsigned redundant_another_pc_loaded;
    unsigned dead_block_misses;
    unsigned i_loaded_hits;
    unsigned other_loaded_hits;
};

bool get_input( const char* config_file_name, simlator_input& sim_input );
void process_stream_access_file( const char* access_file_name,
                                 std::list< access_stream_entry >& stream_list );
void run_simulator( const simlator_input& input );
void run_without_rescheduling( data_cache& simulation_cache,
                               const simlator_input& input,
                               std::list< unsigned >& warp_order,
                               access_stats& stats );
void run_rescheduled( data_cache& simulation_cache, const simlator_input& input, std::list< unsigned >& rescheduled_order );
unsigned bundle_access_stream_by_warp_pc(
        std::list< std::map< unsigned, std::list< std::list< access_stream_entry > > > >&
            warp_to_pc_bundle_to_access_list,
        const std::list< access_stream_entry >& input_access_stream, const simlator_input& input );

int main( int arcg, char* argv[] )
{
    // Step one, parse the configuration file
    simlator_input input;
    if ( get_input( CACHE_CONFIG_FILE, input ) ) {
        g_cache_oracle = new cache_oracle( input.configuration, input.access_stream );
        run_simulator( input );
        delete g_cache_oracle;
    }
    return 0;
}


// Function takes the name of the configuration file and a reference to the simulator_input
// structure which it fills based on the input file.  if parsing fails, it returns false. o.w. true
bool get_input( const char* config_file_name, simlator_input& sim_input )
{
    const char* access_stream_file;
    option_parser_t parser =option_parser_create();
    option_parser_register( parser,
                            "-gpgpu_cache",
                            OPT_CSTR,
                            &sim_input.configuration.m_config_string,
                            "The configuration for the cache.",
                            "N,64:128:6:L:R:m,A:16:4,4:0:0:0:0" );
    option_parser_register( parser,
                            "-access_stream_file",
                            OPT_CSTR,
                            &access_stream_file,
                            "The configuration for the cache.",
                            "none" );
    option_parser_register( parser,
                            "-enable_cache_flushes",
                            OPT_UINT32,
                            &sim_input.enable_flushes,
                            "Flush the cache when the access stream says so?",
                            "1" );
    option_parser_register( parser,
                            "-access_rescheduler",
                            OPT_UINT32,
                            &sim_input.access_rescheduler,
                            "access_rescheduler_type -> changes the access stream based on some criteria",
                            "0" );
    option_parser_register( parser,
                            "-do_not_cache_dead_blocks",
                            OPT_UINT32,
                            &sim_input.do_not_cache_dead_blocks,
                            "look head in the access stream and do not cache any blocks that are not re-referenced anymore",
                            "0" );
    option_parser_register( parser,
                            "-do_not_evict_global_writes",
                            OPT_UINT32,
                            &sim_input.do_not_evict_global_writes,
                            "do not access the cache on a global write",
                            "0" );
    option_parser_cfgfile( parser, CACHE_CONFIG_FILE );

    printf( "cache_sim::config=%s\n", sim_input.configuration.m_config_string );
    printf( "cache_sim::access_stream_file=%s\n", access_stream_file );

    sim_input.configuration.init();
    process_stream_access_file( access_stream_file, sim_input.access_stream );

    // You cannot have an optimal replacement policy and a rescheduled access stream
    // If you want a belady optimal replacement on a rescheduled stream, the you must output the
    // access stream of a rescheduled run then run it again with belady.
    if( BELADY_OPTIMAL == sim_input.configuration.get_replacement_policy() ) {
        assert( ACCESS_RESCHEDULER_NONE == sim_input.access_rescheduler );
    }

    option_parser_destroy( parser );
    return true;
}

void process_stream_access_file( const char* access_file_name,
                                 std::list< access_stream_entry >& stream_list )
{
    FILE* access_list = fopen( access_file_name, "r" );
    fseek( access_list, 0L, SEEK_END );
    size_t file_size = ftell( access_list );
    fseek( access_list, 0L, SEEK_SET );
    char* file_characters = (char*)malloc( file_size );
    size_t bytes_read = fread( file_characters, 1, file_size, access_list );
    assert( file_size == bytes_read );

    std::vector<std::string> strs;
    boost::split(strs, file_characters, boost::is_any_of("\n"));

    for ( std::vector<std::string>::const_iterator iter = strs.begin();
            iter != strs.end(); ++ iter ) {
        if ( strlen( iter->c_str() ) > 0 ) {
            stream_list.push_back( access_stream_entry( iter->c_str() ) );
        }
    }

    free( file_characters );
    fclose( access_list );
}

void run_simulator( const simlator_input& input ) {
    data_cache simulation_cache( "cache_sim_cache",
                                 input.configuration,
                                 0,
                                 0,
                                 NULL,
                                 NULL,
                                 IN_L1D_MISS_QUEUE,
                                 false );
    std::list< unsigned > warp_order;
    access_stats stats;
    if ( ACCESS_RESCHEDULER_NONE == input.access_rescheduler ) {
        run_without_rescheduling( simulation_cache, input, warp_order, stats );
    } else {
        run_rescheduled( simulation_cache, input, warp_order );
    }

    unsigned previously_scheduled = 0xDEADBEEF;
    unsigned num_times_scheduled = 0;
    printf( "warp ordering:\n" );
    for ( std::list< unsigned >::const_iterator iter = warp_order.begin();
            iter != warp_order.end(); ++iter ) {
        if ( *iter == previously_scheduled ) {
            ++num_times_scheduled;
        } else {
            if ( previously_scheduled != 0xDEADBEEF ) {
                printf( "%u - %u times\n", previously_scheduled, num_times_scheduled );
            }
            num_times_scheduled = 0;
        }
        previously_scheduled = *iter;
    }
    if ( num_times_scheduled > 0 ) {
        printf( "%u - %u times\n", previously_scheduled, num_times_scheduled );
    }

    unsigned accesses = 0, misses = 0;
    simulation_cache.print( stdout, accesses, misses );
    printf( "cache_sim::accesses_someone_else_initially_loaded=%f\n",
            (float)stats.accesses_someone_else_initially_loaded / (float)accesses );
    printf( "cache_sim::accesses_i_initially_loaded=%f\n",
            (float)stats.accesses_i_initially_loaded / (float)accesses );
    printf( "cache_sim::access_nobody_touched_yet=%f\n",
            (float)stats.access_nobody_touched_yet / (float)accesses );
    printf( "cache_sim::writes_to_block_previously_loaded=%u\n",
            stats.writes_to_block_previously_loaded );
    printf( "cache_sim::writes_to_previouly_unreferrenced_blocks=%u\n",
            stats.writes_to_previouly_unreferrenced_blocks );
    printf( "cache_sim::redundant_this_pc_loaded=%u\n", stats.redundant_this_pc_loaded );
    printf( "cache_sim::redundant_another_pc_loaded=%u\n", stats.redundant_another_pc_loaded );
    printf( "cache_sim::dead_block_misses=%u\n", stats.dead_block_misses );
    printf( "cache_sim::i_loaded_hits=%u\n", stats.i_loaded_hits );
    printf( "cache_sim::other_loaded_hits=%u\n", stats.other_loaded_hits );
    printf( "cache_sim::accesses=%u\n", accesses );
    printf( "cache_sim::misses=%u\n", misses );
    printf( "cache_sim::miss_rate=%f\n", (float)( misses + stats.dead_block_misses ) /(float)( accesses + stats.dead_block_misses ) );
}

void run_without_rescheduling( data_cache& simulation_cache,
                               const simlator_input& input,
                               std::list< unsigned >& warp_order,
                               access_stats& stats )
{
    unsigned iter_num = 0;
    std::map< new_addr_type, std::list< unsigned > > address_to_warp_touching_list;
    std::map< new_addr_type, std::list< unsigned > > address_to_pc_touching_list;
    for ( g_access_stream_iterator = input.access_stream.begin();
            g_access_stream_iterator != input.access_stream.end(); ++g_access_stream_iterator, ++iter_num ) {
        if ( g_access_stream_iterator->is_cache_flush() ) {
            if ( input.enable_flushes ) {
                simulation_cache.flush();
                address_to_warp_touching_list.clear();
                address_to_pc_touching_list.clear();
            }
        } else {
            const bool is_write = g_access_stream_iterator->get_mem_space_type() == GLOBAL_ACC_W ||
                    g_access_stream_iterator->get_mem_space_type() == LOCAL_ACC_W;
            mem_fetch mf(
                    mem_access_t( g_access_stream_iterator->get_mem_space_type() ,
                                  g_access_stream_iterator->get_addr(),
                                  input.configuration.get_line_sz(),
                                  is_write ),
                                  NULL, 0, 0, 0, 0, NULL );
            std::list<cache_event> events;
            const new_addr_type block_addr = input.configuration.block_addr( g_access_stream_iterator->get_addr() );
            if ( input.do_not_cache_dead_blocks ) {
                if ( g_cache_oracle->get_future_access_position( block_addr, g_access_stream_iterator )
                        == cache_oracle::NEVER_REREFERENCED && simulation_cache.probe( block_addr ) == MISS && !is_write ) {
                    ++stats.dead_block_misses;
                    continue;
                }
            }
            if ( input.do_not_evict_global_writes && g_access_stream_iterator->get_mem_space_type() == GLOBAL_ACC_W ) {
                continue;
            }
            cache_request_status status = simulation_cache.access( g_access_stream_iterator->get_addr(),
                    &mf, iter_num, events );

            // Warp touching stats
            //--------------------------------------------------------------------------------------
            if ( address_to_warp_touching_list[ block_addr ].size() > 0 ) {
                if ( is_write ) {
                    if ( g_access_stream_iterator->get_mem_space_type() == GLOBAL_ACC_W ) {
                        address_to_warp_touching_list[ block_addr ].clear();
                    }
                    ++stats.writes_to_block_previously_loaded;
                } else if ( address_to_warp_touching_list[ block_addr ].front() == g_access_stream_iterator->get_wid() ) {
                    stats.accesses_i_initially_loaded++;
                    if ( status == HIT ) {
                        ++stats.i_loaded_hits;
                    }
                } else {
                    stats.accesses_someone_else_initially_loaded++;
                    if ( status == HIT ) {
                        ++stats.other_loaded_hits;
                    }
                }
            } else {
                if ( is_write ) {
                    ++stats.writes_to_previouly_unreferrenced_blocks;
                } else {
                    stats.access_nobody_touched_yet++;
                }
            }
            address_to_warp_touching_list[ block_addr ].push_back( g_access_stream_iterator->get_wid() );
            //--------------------------------------------------------------------------------------

            // PC touching stats
            //--------------------------------------------------------------------------------------
            if ( address_to_pc_touching_list[ block_addr ].size() > 0 ) {
                if ( is_write && g_access_stream_iterator->get_mem_space_type() == GLOBAL_ACC_W ) {
                    address_to_pc_touching_list[ block_addr ].clear();
                } else if ( address_to_pc_touching_list[ block_addr ].front() == g_access_stream_iterator->get_pc() ) {
                    stats.redundant_this_pc_loaded++;
                } else {
                    stats.redundant_another_pc_loaded++;
                }
            }
            address_to_pc_touching_list[ block_addr ].push_back( g_access_stream_iterator->get_pc() );
            //--------------------------------------------------------------------------------------

            assert( status != RESERVATION_FAIL );
            simulation_cache.instant_fill_cycle( iter_num );
        }
        if ( iter_num % 100 == 0 ) {
            unsigned accesses = 0, misses = 0;
            printf( "iter_num=%u\n", iter_num );
            simulation_cache.print( stdout, accesses, misses );
        }
        warp_order.push_back( g_access_stream_iterator->get_wid() );
    }
}

void run_rescheduled( data_cache& simulation_cache, const simlator_input& input, std::list< unsigned >& rescheduled_order )
{
    const unsigned num_accesses = bundle_access_stream_by_warp_pc( g_warp_to_pc_bundle_to_access_list, input.access_stream, input );
    printf( "num_accesses=%u\n", num_accesses );
    switch( input.access_rescheduler ) {
    case ACCESS_RESCHEDULER_WARP_WITH_MOST_IMMEDIATE_HITS: // Intentional fall through
    case ACCESS_RESCHEDULER_WARP_WITH_LEAST_IMMEDIATE_MISSES: // Intentional fall through
    case ACCESS_RESCHEDULER_WARP_WITH_HIGHEST_IMMEDATE_HIT_RATE: // Intentional fall through
    case ACCESS_RESCHEDULER_WARP_WHERE_MOST_LANES_HIT:
    case ACCESS_RESCHEDULER_CONCURRENCY_LIMITED: {
        unsigned iter_num = 0;
        while ( iter_num < num_accesses ) {
            unsigned max_hits = 0;
            unsigned least_misses = (unsigned)-1;
            float highest_hit_rate = 0.0f;
            unsigned most_lanes_hit =0;
            unsigned warp_with_most_hits = 0, warp_with_fewest_misses = 0, warp_with_highest_hit_rate = 0, warp_with_most_lanes_hit = 0;
            for ( std::map< unsigned, std::list< std::list< access_stream_entry > > >::const_iterator iter
                    = g_warp_to_pc_bundle_to_access_list.front().begin();
                    iter != g_warp_to_pc_bundle_to_access_list.front().end(); ++iter ) {
                unsigned num_hits = 0;
                unsigned num_misses = 0;
                unsigned curr_lanes_hit = 0;
                if ( iter->second.size() > 0 ) {
                    for ( std::list< access_stream_entry >::const_iterator iter2 = iter->second.front().begin();
                          iter2 != iter->second.front().end(); ++iter2 ) {
                        if ( simulation_cache.probe( iter2->get_addr() ) == HIT ) {
                            ++num_hits;
                            curr_lanes_hit += iter2->get_num_lanes_touching();
                        }
                        if ( simulation_cache.probe( iter2->get_addr() ) == MISS ) {
                            num_misses++;
                        }
                    }
                }
                if ( num_hits > max_hits ) {
                    max_hits = num_hits;
                    warp_with_most_hits = iter->first;
                }
                if ( num_misses < least_misses ) {
                    least_misses = num_misses;
                    warp_with_fewest_misses = iter->first;
                }
                const float curr_hit_rate = ( float )( num_hits ) / ( float )( num_misses + num_hits );
                if ( curr_hit_rate > highest_hit_rate ) {
                    highest_hit_rate = curr_hit_rate;
                    warp_with_highest_hit_rate = iter->first;
                }
                if ( curr_lanes_hit > most_lanes_hit  ) {
                    most_lanes_hit = curr_lanes_hit;
                    warp_with_most_lanes_hit = iter->first;
                }
            }
            unsigned selected_warp = input.access_rescheduler == ACCESS_RESCHEDULER_WARP_WITH_MOST_IMMEDIATE_HITS ? warp_with_most_hits
                    : input.access_rescheduler == ACCESS_RESCHEDULER_WARP_WITH_LEAST_IMMEDIATE_MISSES ? warp_with_fewest_misses
                    : input.access_rescheduler == ACCESS_RESCHEDULER_WARP_WITH_HIGHEST_IMMEDATE_HIT_RATE ? warp_with_highest_hit_rate
                    : input.access_rescheduler == ACCESS_RESCHEDULER_WARP_WHERE_MOST_LANES_HIT ? warp_with_most_lanes_hit
                    : 0;

            unsigned num_empty = 0;
            while( g_warp_to_pc_bundle_to_access_list.front()[ selected_warp ].size() == 0 ) {
                ++num_empty;
                selected_warp = ( selected_warp + 1 ) % g_warp_to_pc_bundle_to_access_list.front().size();
                if ( num_empty == g_warp_to_pc_bundle_to_access_list.front().size() ) {
                    printf("FLUSH\n");
                    rescheduled_order.push_back( 777777 );
                    g_warp_to_pc_bundle_to_access_list.pop_front();
                    if ( input.enable_flushes ) {
                        simulation_cache.flush();
                    }
                    continue;
                }
            }
            rescheduled_order.push_back( selected_warp );
            std::list< access_stream_entry >& access
                = g_warp_to_pc_bundle_to_access_list.front()[ selected_warp ].front();
            for ( std::list< access_stream_entry >::const_iterator iter = access.begin();
                    iter != access.end(); ++iter ) {
                mem_fetch mf(
                mem_access_t( iter->get_mem_space_type() ,
                              iter->get_addr(),
                              input.configuration.get_line_sz(),
                              iter->get_mem_space_type() == GLOBAL_ACC_W ||
                              iter->get_mem_space_type() == LOCAL_ACC_W ),
                              NULL, 0, 0, 0, 0, NULL );
                std::list<cache_event> events;
                cache_request_status status = simulation_cache.access( iter->get_addr(),
                        &mf, iter_num, events );
                assert( status != RESERVATION_FAIL );
                simulation_cache.instant_fill_cycle( iter_num );
                ++iter_num;
                if ( iter_num % 100 == 0 ) {
                    unsigned accesses = 0, misses = 0;
                    printf( "iter_num= %u/%u = %.2f%%\n", iter_num, num_accesses, (float)(iter_num)/(float)num_accesses * 100.0f );
                    simulation_cache.print( stdout, accesses, misses );
                }
            }
            g_warp_to_pc_bundle_to_access_list.front()[ selected_warp ].pop_front();
        }
    } break;
    default:
        fprintf( stderr, "Error - unknown input.access_rescheduler - %u\n", input.access_rescheduler );
        abort();
    }
}

#define DEBUG_BUNDLING 0
unsigned bundle_access_stream_by_warp_pc(
        std::list< std::map< unsigned, std::list< std::list< access_stream_entry > > > >&
            list_of_warp_to_pc_bundle_to_access_list,
        const std::list< access_stream_entry >& input_access_stream, const simlator_input& input )
{
    unsigned num_accesses = 0;
    const unsigned INVALID_WARP = (unsigned)-1;
    const new_addr_type INVALID_PC = 0xDEADBEEF;
    unsigned last_warp = INVALID_WARP;
    new_addr_type last_pc =  INVALID_PC;
    std::map< unsigned, std::list< std::list< access_stream_entry > > > warp_to_pc_bundle_to_access_list;
    for ( std::list< access_stream_entry >::const_iterator iter = input.access_stream.begin();
            iter != input.access_stream.end(); ++iter ) {
        if ( iter->is_cache_flush() ) {
            if ( input.enable_flushes && warp_to_pc_bundle_to_access_list.size() > 0 ) {
                list_of_warp_to_pc_bundle_to_access_list.push_back( warp_to_pc_bundle_to_access_list );
                warp_to_pc_bundle_to_access_list.clear();
                last_warp = INVALID_WARP;
                last_pc =  INVALID_PC;
                continue;
            }
        } else {
            if ( iter->get_wid() == last_warp && iter->get_pc() == last_pc ) {
                warp_to_pc_bundle_to_access_list[ iter->get_wid() ].back().push_back( *iter );
            } else {
                std::list< access_stream_entry > new_list;
                new_list.push_back( *iter );
                warp_to_pc_bundle_to_access_list[ iter->get_wid() ].push_back( new_list );
            }
            ++num_accesses;
        }
        last_warp = iter->get_wid();
        last_pc = iter->get_pc();
    }
    if ( !input.enable_flushes ) {
        list_of_warp_to_pc_bundle_to_access_list.push_back( warp_to_pc_bundle_to_access_list );
    }
#if DEBUG_BUNDLING
    for( std::map< unsigned, std::list< std::list< access_stream_entry > > >::const_iterator iter =
                warp_to_pc_bundle_to_access_list.begin(); iter != warp_to_pc_bundle_to_access_list.end();
            ++ iter ) {
        printf( "Warp %u:\n", iter->first );
        for ( std::list< std::list< access_stream_entry > >::const_iterator iter2 = iter->second.begin();
                iter2 != iter->second.end(); ++iter2 ) {
            printf( "\tPC %llu:\n", iter2->begin()->get_pc() );
            printf( "\t\t" );
            for( std::list< access_stream_entry >::const_iterator iter3 = iter2->begin();
                    iter3 != iter2->end(); ++iter3 ) {
                iter3->print( stdout, "," );
            }
            printf( "\n" );
        }
    }
#endif
    return num_accesses;
}
