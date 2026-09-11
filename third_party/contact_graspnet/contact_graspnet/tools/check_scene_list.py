from graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp

grasp_root = '/aihub/OccludedObjectDataset/others/graspnet_1billion/'
camera = 'realsense'


test_seen_objects = [0, 2, 5, 7, 8, 9, 11, 14, 17, 18, 20, 21, 22, 26, 27, 29, 30, 38, 41, 48, 51, 52, 58, 60, 61, 62, 63, 66]
test_similar_objects = [1, 3, 4, 6, 10, 12, 13, 16, 19, 23, 25, 35, 39, 42, 50, 53, 54, 59, 64, 65, 67, 68]
test_novel_objects = [3, 4, 6, 17, 19, 20, 24, 26, 27, 28, 30, 31, 32, 33, 35, 45, 47, 49, 51, 55, 62, 63, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]
grasp_clutter_6d_objects = [42, 47, 50, 53, 54, 67, 60, 62, 63, 64, 65, 45, 70, 74, 33, 37, 59] + list(range(75, 88))


g = GraspNet(root=grasp_root, camera=camera, split='all')
scene_ids = g.getSceneIds()
total_num_non_overlap = 0
total_num_overlap = 0
for scene_id in scene_ids:
    if scene_id < 160:
        continue
    scene_objects = g.getObjIds(scene_id)
    # if len(set(test_seen_objects) & set(scene_objects)) > 2:
    #     continue
    # if len(set(grasp_billion_objects) & set(scene_objects)) > 2:
        # continue
    num_non_overlap = len(scene_objects) - len(set(grasp_clutter_6d_objects) & set(scene_objects))
    num_overlap = len(set(grasp_clutter_6d_objects) & set(scene_objects))
    
    total_num_non_overlap += num_non_overlap
    total_num_overlap += num_overlap
    print(scene_id, num_non_overlap, num_overlap)
print(total_num_non_overlap, total_num_overlap)
    
    
    