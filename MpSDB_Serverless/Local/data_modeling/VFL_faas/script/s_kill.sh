#!/bin/bash

# Terminate data servers
kill $(cat pid/record_1.txt)
kill $(cat pid/record_2.txt)
kill $(cat pid/record_3.txt)
# kill $(cat pid/launch_mlp_label_owner.txt)

