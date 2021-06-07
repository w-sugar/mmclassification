_base_ = [
    '../_base_/models/resnet50_visdrone.py', '../_base_/datasets/visdrone_bs32.py',
    '../_base_/schedules/visdrone_bs256.py', '../_base_/default_runtime.py'
]
