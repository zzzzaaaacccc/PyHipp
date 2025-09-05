#!/bin/bash

echo "Number of hkl files"
find . -name "*.hkl" | grep -v -e spiketrain -e mountains | wc -l

echo "Number of mda files"
find mountains -name "firings.mda" | wc -l

echo "Start Times"
for f in *.out; do
  echo "==> $f <=="
  grep -m1 "time.struct_time" "$f"
done

echo "End Times"
for f in *.out; do
  echo "==> $f <=="
  last=$(grep -n "time.struct_time" "$f" | tail -n1 | cut -d: -f1)
  if [ -n "$last" ]; then
    sed -n "${last}, \$p" "$f"
  fi
done

