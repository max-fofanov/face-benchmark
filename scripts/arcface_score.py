import cv2
import numpy as np
from insightface.app import FaceAnalysis


app = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

face = app.get(cv2.imread("/path/to/target/image"))

tmp_face = app.get(cv2.imread("/path/to/test/image"))

embeddings1 = face[0].normed_embedding
embeddings2 = tmp_face[0].normed_embedding

similarity = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
print(f"Similarity: {similarity}")
