# 2020, 2023, Jadis Project
# Coded by Remi Petitpierre https://github.com/RPetitpierre
# For Bibliothèque nationale de France (BnF) and EPFL, Swiss Federal Institute of Technology in Lausanne

import numpy as np
import pandas as pd
import cv2, os, tqdm, glob, json
import overpass
from PIL import Image, ImageDraw

from scipy.spatial.distance import euclidean
from matplotlib.tri import Triangulation
from scipy.ndimage.interpolation import map_coordinates

Image.MAX_IMAGE_PIXELS = 500000000


def createAnchorMap(settings: dict, anchor_path: tuple):
    
    def requestOSMAPI(city_name: str, osm_geoloc_maxsize: int, admin_level: int):
    
        api = overpass.API(timeout=3600)
        data = api.get('area[name="{0}"]["admin_level"="{1}"];way(area)[highway];out geom;'.format(
            city_name, str(admin_level)), 
            responseformat = "json")

        bounds, highway_type, ids, geometry = [], [], [], []
        for element in data['elements']:
            try:
                bounds.append(element['bounds'])
                highway_type.append(element['tags']['highway'])
                ids.append(element['id'])
                geometry.append(element['geometry'])
            except:
                pass

        df = pd.DataFrame(bounds)
        df['id'] = ids
        df['highway_type'] = highway_type
        df['geometry'] = geometry

        # Eliminate homonym cities
        for i in range(2):
            df = df[(df['minlat'] > df['minlat'].mean()-(2*df['minlat'].std())) &
                    (df['minlon'] > df['minlon'].mean()-(2*df['minlon'].std())) &
                    (df['maxlat'] < df['maxlat'].mean()+(2*df['maxlat'].std())) &
                    (df['maxlon'] < df['maxlon'].mean()+(2*df['maxlon'].std()))]


        highway_types = {'footway': 1, 'residential': 2, 'service': 2, 'steps': 1, 'pedestrian': 2, 'primary': 4,
             'secondary': 3, 'tertiary': 2, 'cycleway': 1, 'path': 1, 'trunk_link': 2, 'living_street': 2,
             'trunk': 3, 'track': 2, 'primary_link': 2, 'corridor': 1, 'construction': 1,
             'motorway_link': 2, 'secondary_link': 2, 'tertiary_link': 1, 'motorway': 4, 'road': 2}

        df['width'] = 1
        for type_ in highway_types.keys():
            df['width'].loc[df['highway_type'] == type_] = 1 + highway_types[type_]

        return df
    
    
    def drawAnchorMap(anchor: np.ndarray, df: pd.DataFrame, coef: float, streetwidth_coef: float):
    
        def drawStreet(draw: ImageDraw.ImageDraw, geometry: list, coef: float, width: int):

            street = pd.DataFrame(geometry)
            street['lon'] = coef*(street['lon']-minlon)
            street['lat'] = coef*(street['lat']-minlat)

            points = np.ravel(np.asarray((street['lon'].tolist(), street['lat'].tolist())).T).tolist()
            draw.line(points, fill = 255, width = width)

        minlat, maxlat = df['minlat'].min(), df['maxlat'].max()
        minlon, maxlon = df['minlon'].min(), df['maxlon'].max()

        anchor = Image.fromarray(anchor)

        draw = ImageDraw.Draw(anchor)
        fdrawStreet = np.vectorize(drawStreet)
        fdrawStreet(draw, df['geometry'].values, coef, df['width'].values*streetwidth_coef)
        del draw

        anchor = anchor.transpose(Image.FLIP_TOP_BOTTOM)
        anchor.save('workshop/PIL.png')

        return anchor

    save_path = os.path.join(*anchor_path, 'projection.json')

    if False:#len(glob.glob(save_path)) > 0:
        print("\nThe anchor has already been calculated. You will find it at the following location: {0}\n".format(save_path))

    else:    
        df = requestOSMAPI(settings['corpus']['city_name'], settings['anchor']['image_maxsize'], settings['anchor']['admin_level'])

        minlat, maxlat = df['minlat'].min(), df['maxlat'].max()
        minlon, maxlon = df['minlon'].min(), df['maxlon'].max()

        h, w = maxlat-minlat, maxlon-minlon
        coef = settings['anchor']['image_maxsize']/np.max([h, w])

        anchor = np.zeros((int(np.around(h*coef)), int(np.around(w*coef)))).astype('uint8')

        anchor = drawAnchorMap(anchor, df, coef, settings['anchor']['streetwidth_coef'])

        citylat = np.mean([minlat, maxlat])
        if maxlat > 0:
            top_lon_deformation = np.cos(np.pi*maxlat/180)
            bot_lon_deformation = np.cos(np.pi*minlat/180)
        else:
            top_lon_deformation = np.cos(np.pi*minlat/180)
            bot_lon_deformation = np.cos(np.pi*maxlat/180)

        lon_deformation = np.max([top_lon_deformation, bot_lon_deformation])
        
        anchor = np.array(anchor)
        anchor = cv2.resize(anchor, (int(np.around(anchor.shape[1]*lon_deformation)), anchor.shape[0])).astype('uint8')   
        pixel_offset = (lon_deformation-top_lon_deformation)*anchor.shape[1]/2
        
        (h, w) = anchor.shape

        X = np.asarray([0, 0, h, h])
        Y = np.asarray([0, w, 0, w])
        Zx = np.asarray([0, 0, 0, 0])
        if top_lon_deformation <  bot_lon_deformation:
            Zy = np.asarray([-pixel_offset, pixel_offset, 0, 0])
        else:
            Zy = np.asarray([0, 0, -pixel_offset, pixel_offset])

        dx, dy = computeDeformation(X, Y, Zx, Zy, (h, w))
        transform = elasticTransform(anchor, dx, dy)

        transform[transform < 255/2] = 0
        transform[transform > 0] = 255 
        
        cv2.imwrite(os.path.join(*anchor_path, 'anchor.png'), transform)
        
        with open(save_path, 'w') as outfile:
            json.dump({'bot_lon_deformation': bot_lon_deformation, 
                       'top_lon_deformation': top_lon_deformation, 
                       'minlon': minlon, 'maxlon': maxlon, 
                       'minlat': minlat, 'maxlat': maxlat, 
                       'coef': coef, 'shape': [h, w]}, outfile)

def loadProjectionParams(anchor_path: tuple):
    ''' Load anchor reprojection parameters '''

    with open(os.path.join(*anchor_path, 'projection.json')) as data:

        data = json.load(data)
        bot_lon_deformation = data['bot_lon_deformation']
        top_lon_deformation = data['top_lon_deformation']
        minlon = data['minlon']
        maxlon = data['maxlon']
        minlat = data['minlat']
        maxlat = data['maxlat']
        coef = data['coef']
        shape = data['shape']
        
    return bot_lon_deformation, top_lon_deformation, minlon, maxlon, minlat, maxlat, coef, shape

def toLatLon(coords, anchor_path: tuple):
    
    bot_lon_deformation, top_lon_deformation, minlon, maxlon, minlat, maxlat, coef, shape = loadProjectionParams(anchor_path)
    
    latitude, longitude = [], []
    lon_def_coef = bot_lon_deformation-top_lon_deformation
    mid_w = shape[1]/2
    mid_lon = np.mean([minlon, maxlon])

    for i in range(len(coords)):
        lat = coords[i, 1]/coef
        lon = coords[i, 0]
                
        if maxlat > 0:
            latitude.append(maxlat-lat)
        else:
            latitude.append(minlat-lat)

        lon_def = top_lon_deformation + lon_def_coef*lat/shape[0]
        longitude.append(mid_lon+((lon - mid_w)/lon_def)/coef)

    geolocation = np.asarray([latitude, longitude]).T
    
    return geolocation


def addFringePoints(src_pts_matched: np.ndarray, errors: np.ndarray, shape: tuple):
    ''' Adds static points along the fringe of the image to avoid deformation of the frame. Then, 
        computes and returns the X and Y deformation for all the points, including fringe points.
    Input(s):
        src_pts_matched: keypoints matched in the source image
        errors: error of the keypoints with regard to the reference map
        shape: shape of the image to deform
    Output(s):
        X: vertical coordinate of the keypoints, including fringe keypoints
        Y: horizontal coordinate of the keypoints, including fringe keypoints
        Zx: vertical deformation of the keypoints
        Zy: horizontal deformation of the keypoints
    '''

    fringe = (int(np.around(shape[0]/100)), int(np.around(shape[1]/100)))
    x = np.linspace(0, shape[0], fringe[0]).tolist()
    x += np.linspace(0, shape[0], fringe[0]).tolist()
    x += np.zeros(fringe[1]).tolist()
    x += (shape[0]*np.ones(fringe[1])).tolist()

    y = np.zeros(fringe[0]).tolist()
    y += (shape[1]*np.ones(fringe[0])).tolist()
    y += np.linspace(0, shape[1], fringe[1]).tolist()
    y += np.linspace(0, shape[1], fringe[1]).tolist()

    x, y = np.asarray(x).astype('int'), np.asarray(y).astype('int')
    z = np.zeros(2*(fringe[0]+fringe[1])).astype('int')
    
    X = np.concatenate((src_pts_matched[:, 1], x))
    Y = np.concatenate((src_pts_matched[:, 0], y))
    Zx = np.concatenate((errors[:, 0], z))
    Zy = np.concatenate((errors[:, 1], z))

    return X, Y, Zx, Zy


def computeErrors(src_pts_matched: np.ndarray, dst_pts_matched: np.ndarray, M: np.ndarray):
    ''' Computes the error, or the deformation between the matched keypoints of both images, 
    with regard to the homography.
    Input(s):
        src_pts_matched: keypoints matched in the source image
        dst_pts_matched: keypoints matched in the destination image
        M: matrix of transformation src->dst
    Output(s):
        errors: error of the keypoints in the second image with regard to the reference '''
    
    errors = dst_pts_matched - ((src_pts_matched@M[:,:2]) + M[:, 2:3].transpose())
    
    errors = np.asarray(errors)
    
    return errors


def computeDeformation(X, Y, Zx, Zy, shape):
    ''' Computes the error, or the deformation between the matched keypoints of both images, 
        at each pixel position in the second image.
    Input(s):
        X: vertical coordinate of the keypoints, including fringe keypoints
        Y: horizontal coordinate of the keypoints, including fringe keypoints
        Zx: vertical deformation of the keypoints
        Zy: horizontal deformation of the keypoints
        shape: shape of the image to deform
    Output(s):
        dx: map of the vertical deformation, at each pixel position in the 2nd image
        dy: map of the horizontal deformation, at each pixel position in the 2nd image
    '''

    triangulation = Triangulation(X, Y)
    finder = triangulation.get_trifinder()

    triangle = np.zeros(shape)
    j_coords = np.arange(shape[1])

    for i in range(shape[0]):
        triangle[i] = finder(i*np.ones(shape[1]).astype('int64'), j_coords) 

    array_x = triangulation.calculate_plane_coefficients(Zx)
    array_y = triangulation.calculate_plane_coefficients(Zy)

    n_triangle = array_x.shape[0]
    dx, dy = np.zeros(shape), np.zeros(shape)
    indices = np.indices(shape)

    dx = indices[0]*array_x[:,0][triangle.astype('int16')] + indices[1]*array_x[:, 1][triangle.astype('int16')] + \
                    array_x[:,2][triangle.astype('int16')]
    dy = indices[0]*array_y[:,0][triangle.astype('int16')] + indices[1]*array_y[:, 1][triangle.astype('int16')] + \
                        array_y[:,2][triangle.astype('int16')]
            
    return dx, dy


def elasticTransform(image: np.ndarray, dx, dy):
    '''Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    '''
    def transformChannel(channel, dx, dy):
        
        x, y = np.meshgrid(np.arange(channel.shape[0]), np.arange(channel.shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        return map_coordinates(channel, indices, order=1).reshape(channel.shape)
        
    if len(image.shape)==2:
        
        return transformChannel(image, dx, dy)
    
    else:
        r, g, b = cv2.split(image)
        r_ = transformChannel(r, dx, dy)
        g_ = transformChannel(g, dx, dy)
        b_ = transformChannel(b, dx, dy)
        
        transformed_image = cv2.merge([r_, g_, b_])
        
        return transformed_image


def saveDeformation(dx: np.ndarray, dy: np.ndarray, name: str, primary: bool = True):
    ''' Converts the dx and dy deformation to an image for storage. Lossy.
    Input(s):
        path: folder to store the deformation
        dx: map of the vertical deformation, at each pixel position in the 2nd image
        dy: map of the horizontal deformation, at each pixel position in the 2nd image
    '''

    project_name = getProjectName()

    if primary:
        export_path = os.path.join('export', project_name, 'deformation', 'primary')
    else:
        export_path = os.path.join('export', project_name, 'deformation', 'secondary')
    
    sign_channel = ((dx > 0) + 2*(dy > 0)).astype('uint8')
    coef = 255/np.max([np.max(np.abs(dx)), np.max(np.abs(dy))])
    if coef > 1:
        coef = 1
        
    dx_ = np.abs(np.around(dx*coef)).astype('uint8')
    dy_ = np.abs(np.around(dy*coef)).astype('uint8')
        
    compact = cv2.merge([dx_, sign_channel, dy_])
    cv2.imwrite(os.path.join(export_path, name + '.png'), compact)
    
    return coef


def loadDeformation(path: str):
    ''' Loads the stored deformation.
    Input(s):
        path: folder where the deformation is stored
        image_name: name of the 2nd image
    Output(s):
        dx: map of the vertical deformation, at each pixel position in the 2nd image
        dy: map of the horizontal deformation, at each pixel position in the 2nd image
    '''
    
    compact = cv2.imread(path)
    dx_, sign_channel, dy_ = cv2.split(compact)
    dx_, dy_ = dx_.astype('int16'), dy_.astype('int16')
    
    dx_[(sign_channel == 0) | (sign_channel == 2)] = -dx_[(sign_channel == 0) | (sign_channel == 2)]
    dy_[(sign_channel == 0) | (sign_channel == 1)] = -dy_[(sign_channel == 0) | (sign_channel == 1)]
    
    return dx_, dy_


def applyDeformation(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, coef: float):
    
    deformation_coef = 1/coef
    
    dx = cv2.resize(dx, (img.shape[1], img.shape[0]))*deformation_coef
    dy = cv2.resize(dy, (img.shape[1], img.shape[0]))*deformation_coef
    
    deformed = elasticTransform(img, dx, dy)
    
    return deformed

def apply_transform(source_pts: np.ndarray, M: np.ndarray):

    # The set of points you want to transform
    points = np.array(source_pts, dtype=np.float32)

    # Convert the points to homogeneous coordinates
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))

    # Multiply the transformation matrix M by the homogeneous coordinates of each point
    transformed_homogeneous_points = np.dot(np.array(M, dtype=np.float32), homogeneous_points.T)

    # Convert the resulting homogeneous coordinates back to Euclidean coordinates
    transformed_points = transformed_homogeneous_points[:2, :] / transformed_homogeneous_points[2, :]
    
    return transformed_points.T


def realign(sg_output_folder: tuple, anchor_folder: tuple,
            anchor_name: str, anchor_scale: int, scale_factor: dict, orig_shape: dict, ransac_radius: float = 50., 
            with_validation: bool = False, control_pts_folder: tuple = None):
    """
    Realign maps using homography, optionally validating the realignment 
    against manually annotated control points.

    Input(s):
        sg_output_folder (tuple of str): path to folder where pre-matched keypoints are stored.
        ransac_radius (float, optional): RANSAC tolerance radius. Defaults to 50.
        with_validation (bool, optional): Whether to validate the realignment. Defaults to False.
        control_pts_folder (tuple of str, optional): path to control points folder (only for validation mode)

    Output(s):
        list of dict: Realignment results.
    """
    
    output = []
        
    for npz_file in tqdm.tqdm(sorted(glob.glob(os.path.join(*sg_output_folder, "*.npz")))):
        
        map_name = os.path.basename(npz_file).split((anchor_name.split('.')[0]))[0][:-1]
        image_scale = scale_factor[map_name]
        shape = orig_shape[map_name]
        npz = np.load(npz_file)
        
        log = {
            'source_map': map_name,
            'dest_map': anchor_name,
            'source_shape': shape,
        }
        
        # Retrieve stored keypoints for both source and destination maps
        src_pts, dst_pts = [], []
        for i in range(len(npz["matches"])):
            if npz["matches"][i] != -1 :
                src_pts.append(npz["keypoints0"][i]*(image_scale))
                dst_pts.append(npz["keypoints1"][npz["matches"][i]]*(anchor_scale))
        src_pts = np.array(src_pts).reshape(-1, 1, 2)
        dst_pts = np.array(dst_pts).reshape(-1, 1, 2)
        
        # Find realignment homography
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_radius)
        except:
            log['success'] = False
            output.append(log)
            continue
            
        corners = toLatLon(apply_transform([[0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]],
                         M), anchor_folder)
        
        log['transform_matrix'] = M.tolist()
        log['lat_lon_corners'] = corners.tolist()
        log['source_pts'] = src_pts[mask[:, 0].astype('bool'), 0].tolist()
        log['dest_pts'] = dst_pts[mask[:, 0].astype('bool'), 0].tolist()
            
        # Validate realignment if required
        if with_validation:
            with open(os.path.join(*control_pts_folder, f'{map_name}.json'), "r") as cp_json:
                test_points = json.load(cp_json)

            pts = np.float32([[x, y] for x, y in zip(test_points["x"], test_points["y"])]).reshape(-1, 1, 2)
            targets = np.float32([[x, y] for x, y in zip(test_points["x_"], test_points["y_"])])

            try:
                p_transformed = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            except:
                log['success'] = False
                output.append(log)
                continue

            res = [np.sqrt((target_x - p_transformed_x)**2 + (target_y - p_transformed_y)**2) for (
                (target_x, target_y), (p_transformed_x, p_transformed_y)) in zip (targets, p_transformed)]
            
            log['target_control_pts'] = targets.tolist()
            log['transformed_control_pts'] = p_transformed.tolist()
            log['residuals'] = res
        
        log['success'] = len(log['source_pts']) >= 10
        output.append(log)

    return output

    