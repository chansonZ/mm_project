#!/bin/bash
read -r -p "Are you sure? [y/N] " response
case $response in
    [yY][eE][sS]|[yY]) 
        rm *.{log,audit};
        rm log/*.log;
        rm data/*.sign*;
        rm -rm .audit_*
        ;;
    *)
        echo "Not doing anything"
        ;;
esac
