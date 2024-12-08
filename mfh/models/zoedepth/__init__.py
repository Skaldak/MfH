from mfh.models.zoedepth.zoedepth_relative import ZoeDepthRelative

all_versions = {
    "relative": ZoeDepthRelative,
}

get_version = lambda v: all_versions[v]
