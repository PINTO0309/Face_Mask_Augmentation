import os
import cv2
import pickle
import numpy as np
from PIL import Image
import random
random.seed(0)

MASK_PTS_FILE = './mask_images/mask_pts.pkl'
TRI_MASK_IDX = [
    [0, 1, 3], [3, 1, 4], [3, 4, 6], [6, 4, 7],
    [4, 7, 8], [4, 5, 8], [1, 5, 4], [1, 2, 5]
]
DEFAULT_TRI_FACE_IDX = [
    [1, 28, 3], [3, 28, 30], [3, 30, 5], [5, 30, 8],
    [30, 8, 11], [30, 13, 11], [28, 13, 30], [28, 15, 13]
]
DEFAULT_MASK_PTS = np.array([
    (30, 12), (125, 5), (220, 12), (20, 80), (125, 80),
    (230, 80), (65, 140), (125, 160), (185, 140)
])


def get_tri_mask_points(pts_mask, tri_mask_idx):
    tri_mask_pts = np.zeros((len(tri_mask_idx), 6), dtype=np.float32)
    for i in range(len(tri_mask_idx)):
        tri_mask_pts[i] = pts_mask[tri_mask_idx[i]].ravel()
    return tri_mask_pts


def closest_point(pt, pts):
    dist = np.sum((pts - pt) ** 2, axis=1)
    return np.argmin(dist), np.min(dist)


def create_mask_mark(png_image):
    create_mask_mark.done = False
    create_mask_mark.current = (0, 0)
    create_mask_mark.pts = DEFAULT_MASK_PTS
    create_mask_mark.sel_idx = None
    window = 'Adjust points'

    def on_mouse(event, x, y, flags, param):
        if create_mask_mark.done:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if create_mask_mark.sel_idx is not None:
                create_mask_mark.pts[create_mask_mark.sel_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            idx, dist = closest_point(np.array((x, y)), create_mask_mark.pts)
            if dist < 10:
                create_mask_mark.sel_idx = idx
        elif event == cv2.EVENT_LBUTTONUP:
            create_mask_mark.sel_idx = None

    masks = []
    idx = []
    if os.path.exists(MASK_PTS_FILE):
        masks = pickle.load(open(MASK_PTS_FILE, 'rb'))
        idx = [i for (i, d) in enumerate(masks) if d['file'] == png_image]
        if len(idx) > 0:
            create_mask_mark.pts = masks[idx[0]]['pts']
    else:
        pass

    img = cv2.imread(png_image, cv2.IMREAD_UNCHANGED)
    cv2.imshow(window, img)
    cv2.waitKey(1)
    cv2.setMouseCallback(window, on_mouse)
    print('Press ESC to finish Adjust.')

    while not create_mask_mark.done:
        canvas = np.copy(img)
        for pt in create_mask_mark.pts:
            canvas = cv2.circle(canvas, (pt[0], pt[1]), 4, (0, 255, 0), -1)

        tri_mask_pts = get_tri_mask_points(create_mask_mark.pts, TRI_MASK_IDX)
        for tri in tri_mask_pts:
            tri = tri.reshape(3, 2)
            canvas = cv2.polylines(canvas, [tri.astype(np.int32)], True, (0, 255, 0), 2)

        cv2.imshow(window, canvas)
        if cv2.waitKey(50) == 27:
            create_mask_mark.done = True

    print('Any KEY to continue.')
    cv2.imshow(window, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(idx) > 0:
        masks[idx[0]]['pts'] = create_mask_mark.pts
    else:
        masks.append({'file': png_image, 'pts': create_mask_mark.pts})
    pickle.dump(masks, open(MASK_PTS_FILE, 'wb'))


class FaceMasker:
    def __init__(self, mask_pts_file=MASK_PTS_FILE):
        self.masks_pts_file = mask_pts_file

        self.num_pts = 9
        self.tri_mask_idx = TRI_MASK_IDX
        self.tri_face_idx = DEFAULT_TRI_FACE_IDX
        self.masks = None
        self.load_mask()

    def load_mask(self):
        masks = pickle.load(open(self.masks_pts_file, 'rb'))

        self.masks = []
        for m in masks:
            img = cv2.imread(m['file'], cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.masks.append(
                {
                    'img': img,
                    'pts': m['pts'],
                    'tri': get_tri_mask_points(m['pts'], self.tri_mask_idx)
                }
            )

    def get_tri_face_points(self, shape_pts):
        tri_face = np.zeros((len(self.tri_face_idx), 6), dtype=np.float32)
        for i in range(len(self.tri_face_idx)):
            for j in range(3):
                pt = shape_pts[self.tri_face_idx[i][j]]
                if hasattr(pt, 'x') and hasattr(pt, 'y'):
                    tri_face[i, [j+j, j+j+1]] = pt.x, pt.y
                else:
                    tri_face[i, [j+j, j+j+1]] = pt[0], pt[1]
        return tri_face

    def wear_mask_to_face(self, image, face_shape, mask_idx=None):
        if mask_idx is None:
            mask_idx = random.randint(0, len(self.masks)-1)

        image_mask = self.masks[mask_idx]['img']
        tri_mask_pts = self.masks[mask_idx]['tri']
        tri_face = self.get_tri_face_points(face_shape)

        image_face = Image.fromarray(image)
        for pts1, pts2 in zip(tri_mask_pts, tri_face):
            pts1 = pts1.copy().reshape(3, 2)
            pts2 = pts2.copy().reshape(3, 2)

            rect1 = cv2.boundingRect(pts1)
            pts1[:, 0] = pts1[:, 0] - rect1[0]
            pts1[:, 1] = pts1[:, 1] - rect1[1]

            croped_tri_mask = image_mask[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]

            rect2 = cv2.boundingRect(pts2)
            pts2[:, 0] = pts2[:, 0] - rect2[0]
            pts2[:, 1] = pts2[:, 1] - rect2[1]

            mask_croped = np.zeros((rect2[3], rect2[2]), np.uint8)
            cv2.fillConvexPoly(mask_croped, pts2.astype(np.int32), 255)

            M = cv2.getAffineTransform(pts1, pts2)
            warped = cv2.warpAffine(croped_tri_mask, M, (rect2[2], rect2[3]))
            warped = cv2.bitwise_and(warped, warped, mask=mask_croped)

            warped = Image.fromarray(warped)
            image_face.paste(warped, (rect2[0], rect2[1]), warped)

        return np.array(image_face)


if __name__ == '__main__':
    create_mask_mark('./mask_images/01_surgical_light_blue.png')
    create_mask_mark('./mask_images/02_cloth.png')
    create_mask_mark('./mask_images/03_surgical_white.png')
    create_mask_mark('./mask_images/04_surgical_blue.png')

    masker = FaceMasker()
