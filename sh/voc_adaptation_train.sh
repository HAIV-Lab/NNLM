echo "Running '$0 $1 $2'..."

usage() {
    echo "Usage:$0 mode task [train_steps] [update_freq]"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

# include const.sh 
# . ./const.sh $mode $task

mode=$1
task=$2
train_steps_specified=$3
_update_freq=$4

if [ -n "$is_valid" ]; then
    echo $is_valid
    exit 1
fi