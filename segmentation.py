from vertex_detect import detect_vertices
from edge_detect import detect_edge




 



if __name__ == "__main__":
    image_path = "test_image/7state_fsm.png"
    nodes = detect_vertices(image_path)
    for node_id, center, radius, label in nodes:
        print(f"Node {node_id}: center {center}, radius {radius}, label: {label}")
    edge = detect_edge(image_path, nodes)
    for edge in edge:
        print(f"Edge: {edge[1]} <- {edge[0]}, label: {edge[2]}")



    


