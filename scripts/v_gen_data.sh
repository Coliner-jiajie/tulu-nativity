#!/bin/bash
# 生成 viz/data.js（JS 引擎，约 1 分钟）。需要先运行过 python scripts/v_export_params.py 生成 viz/params.js。
cd "$(dirname "$0")/../viz" || exit 1
JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
"$JSC" ../scripts/v_gen_data.js > data.js.tmp && mv data.js.tmp data.js && ls -l data.js
