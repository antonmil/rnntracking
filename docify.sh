#!/bin/sh
lua5.1 ~/software/luadoc/src/luadoc.lua.in -d docs \
	src/util/misc.lua \
	src/util/csv.lua \
	src/util/plot.lua \
	src/util/io.lua
