import theano.tensor as tt
import numpy as np
from theano.compile.ops import as_op
from itertools import takewhile

def infer_shape(node, input_shapes):
    ashp, bshp = input_shapes
    return [ashp]

@as_op(itypes=[
                tt.fvector, # x_h_
                tt.fvector, # v_h_
                tt.fvector, # t_h_
                tt.fvector, # turn_vec_h
                tt.fmatrix, # x_t_
                tt.fmatrix, # v_t_
                tt.fvector, # t_t_
                tt.fvector, # turn_vec_t
                tt.bvector, # exist
                tt.fscalar, # time_step
               ],
       otypes=[
                tt.fmatrix, # a_t
                tt.fmatrix, # v_t
                tt.fmatrix, # x_t
                tt.fvector, # t_t
                tt.fvector, # t_h
               ])#, infer_shape=infer_shape)

def step(x_h_, v_h_, t_h_, turn_vec_h, x_t_, v_t_, t_t_, turn_vec_t, exist, time_step):
    '''
    x_h_ : host position in time t-1. (1x2)
    h_h_ : host speed in time t-1. (1x2)
    t_h_ : host waiting time since full stop (1)
    x_t_ : target position in time t-1. (3x2)
    v_t_ : target speed in time t-1. (3x2)
    t_t_ :
            -1 : still approaching stopping line (3)
            >0 : The time i arrived at the junction
            0 : entered the junction
    exist : boolean flag indicating if a car exist in the lane
    '''

    DT = 0.1

    HOST_L    = 144.0 # ???
    HOST_W     = 78.0
    PIX2METER = -5/HOST_L # convert to pixels to meters
    HOST_W_METERS     = -HOST_W*PIX2METER
    HOST_L_METERS     = -HOST_L*PIX2METER
    LANE_WIDTH_FACTOR = 1.3 #FACTOR
    LANE_WIDTH_PX = HOST_L * LANE_WIDTH_FACTOR
    LANE_WIDTH_METERS = -LANE_WIDTH_PX * PIX2METER
    JUNCTION_SIDE_METERS = 2 * LANE_WIDTH_METERS
    TILE_METERS = JUNCTION_SIDE_METERS/2
    FPS = 9 # frames per second
    START_BOTTOM = 0
    START_RIGHT = 1
    START_TOP = 2
    START_LEFT = 3
    STEER_RIGHT = 0
    STEER_STRAIGHT = 1
    STEER_LEFT = 2
    MOVING_SPEED = 3

    # v_t = v_t_ + 1e-6
    # x_t = x_t_ + v_t * DT
    # t_t = t_t_ + DT
    # t_h = t_h_ + DT
    # a_t = np.zeros((3,2),dtype=np.float32)

    state_ = np.zeros(shape=(4,7), dtype=np.float32)
    state_[:-1,:2] = x_t_
    state_[:-1,2] = abs(v_t_).sum(axis=1)
    state_[:-1,4] = np.asarray([1,2,3])
    state_[:-1,5] = turn_vec_t
    state_[:-1,6] = t_t_

    # concatenate host in last row
    state_[-1,:2] = x_h_
    state_[-1,2] = abs(v_h_).sum()
    state_[-1,4] = 0.
    state_[-1,5] = np.nonzero(turn_vec_h)[0]
    state_[-1,6] = t_h_

    x_t = np.zeros_like(x_t_)
    v_t = np.zeros_like(v_t_)
    t_t = np.zeros_like(t_t_)
    t_h = t_h_ + 1
    a_t = np.zeros_like(v_t_)

    for i in range(3):

        #=======================================================================
        # Skip non existing cars
        #=======================================================================
        if exist[i] == 0:
            x_t[i] = x_t_[i]
            v_t[i] = v_t_[i]
            continue

        stete_r = np.matrix.copy(state_)
        #=======================================================================
        # Get target measurments
        #=======================================================================
        my_id = i
        speed = stete_r[my_id][2]
        stop_arrival = stete_r[my_id][6]
        steer = stete_r[my_id][5]
        source = int(stete_r[my_id][4])

        #=======================================================================
        # Translate state according source street
        #=======================================================================
        for _ in xrange(source):
            for r in stete_r:
                t_X = r[0]
                t_Y = r[1]
                r[0], r[1] = t_Y,-t_X
                r[4] = (r[4] - 1)%4

        X_trans = stete_r[my_id][0]
        Y_trans = stete_r[my_id][1]

        dy = speed*DT

        #=======================================================================
        # Movement decision block
        #=======================================================================

        #Before stop line
        if (stop_arrival < 0):
            #Still approaching
            if (Y_trans + dy + 0.4 * HOST_L_METERS < -TILE_METERS):
                Y_trans = Y_trans + dy

            #Set arrival
            else:
                stop_arrival = time_step
                speed = 0
                t_t[i] = stop_arrival

        #Steer
        elif (time_step - stop_arrival) > DT*FPS*3:
            #Check rules here: http://www.vdriveusa.com/resources/how-a-4-way-stop-works.php
            on_track = stop_arrival == 0
            allowed = empty = False
            allowed_vehicle_id = None

            #===================================================================
            # Decicde next allowed vehicle on junction
            #===================================================================
            arrival_times = sorted([(i,t[6]) for i,t in enumerate(stete_r) if t[6] > 0],key=lambda el:el[1])
            next_to_go = [i for i,t in takewhile(lambda (x,y): y<=arrival_times[0][1], arrival_times)]

            if not len(next_to_go): allowed_vehicle_id = my_id
            elif len(next_to_go) == 1: allowed_vehicle_id = next_to_go[0]
            else:
                #State is translated, so it's safe to ask about absolute STATE_RIGHT
                vehicle_ids_on_right = filter(lambda tid: stete_r[tid][4] == START_RIGHT, next_to_go)
                if vehicle_ids_on_right:
                    if steer != STEER_RIGHT:
                        allowed_vehicle_id = vehicle_ids_on_right[0]
                else:
                    vehicle_ids_on_straight = filter(lambda tid: stete_r[tid][4] == START_TOP, next_to_go)
                    if vehicle_ids_on_straight:
                        if steer == STEER_LEFT:
                            vehicle_id_on_straight = vehicle_ids_on_straight[0]
                            is_turning_left = stete_r[vehicle_id_on_straight][5] == STEER_LEFT
                            if is_turning_left:
                                #Dead lock - we both turning left
                                #Check if he's waiting for someone on his right (our left)
                                vehicles_on_his_right = filter(lambda t: t[4] == START_LEFT and t[6] > 0, stete_r)
                                if vehicles_on_his_right:
                                    allowed_vehicle_id = my_id #He's waiting for someone on his left - And I don't - I'll take the lead
                                else:
                                    allowed_vehicle_id = vehicle_id_on_straight #TODO: Dead lock on same street - same time - left
                            elif stete_r[my_id][6] == stete_r[vehicle_id_on_straight][6]:
                                allowed_vehicle_id = my_id #He's waiting for someone on his left - And I don't - I'll take the lead
                            else:
                                allowed_vehicle_id = vehicle_id_on_straight

            if allowed_vehicle_id is None: allowed_vehicle_id = my_id

            #===================================================================
            # Check If it's my turn
            #===================================================================
            if allowed_vehicle_id == my_id:
                allowed = True

            #===================================================================
            # Is the junction empty?
            #===================================================================
            for i, t in enumerate(stete_r):
                if i == my_id: continue
                x,y = t[0], t[1]
                #Get vehicle body
                if source in (START_BOTTOM, START_TOP):
                    top, bottom, left, right =  (y + HOST_L_METERS/3, y - HOST_L_METERS/3, x - HOST_W_METERS/3, x + HOST_W_METERS/3)
                top, bottom, left, right = (y + HOST_W_METERS/3, y - HOST_W_METERS/3, x - HOST_L_METERS/3, x + HOST_L_METERS/3)

                e = speed * DT

                if not ((left - e > TILE_METERS or -TILE_METERS > right + e) or (bottom - e > TILE_METERS or -TILE_METERS > top + e)):
                    break
            else:#Exhaustion
                empty = True

            can_progress = (allowed & empty)

            #===================================================================
            # Do steering
            #===================================================================
            speed = MOVING_SPEED
            dx = speed*DT
            dy = speed*DT

            if steer == STEER_STRAIGHT:
                if on_track or can_progress:
                    Y_trans = Y_trans + dy
                    v_t[i] = np.asarray([0, speed],dtype=np.float32)

            elif steer == STEER_RIGHT:
                if on_track or can_progress:
                    Y_trans = min(Y_trans + dy,-0.5 * TILE_METERS)
                    X_trans = X_trans + dx
                    v_t[i] = np.asarray([speed, 0],dtype=np.float32)

            elif steer == STEER_LEFT:
                if on_track or can_progress:
                    Y_trans = min(Y_trans+dy, 0.5 * TILE_METERS)
                    X_trans = (X_trans - dx) if Y_trans > 0 else X_trans
                    v_t[i] = np.asarray([-speed, 0],dtype=np.float32)

            if can_progress and stop_arrival > 0:
                stop_arrival = 0 #Indicates we moved forward
                t_t[i] = stop_arrival

        #=======================================================================
        # Restore coordinate system
        #=======================================================================
        for src in range(4,source,-1):
            X_trans, Y_trans = Y_trans,-X_trans

        x_t[i] =  [X_trans, Y_trans]
        a_t[i] = v_t[i] - v_t_[i]

    return a_t, v_t, x_t, t_t, t_h

# x = tt.fmatrix()
# # y = tt.fmatrix()
# #
# f = function([x], my_2pir_jump(x))
# inp1 = 160 * np.random.randn(2,1).astype('float32')
# # inp2 = np.random.rand(2,2).astype('float32')
# out = f(inp1)
# print 'inp1:'
# print inp1
# # print 'inp2:'
# # print inp2
# print 'out:'
# print out