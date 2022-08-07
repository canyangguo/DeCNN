import math


EARTH_RADIUS = 6371.004

def rad(d):
    return d * math.pi / 180.0


def dis(X, Y):
    S = 0
    for i in range(X.shape[0]):
        lat1, lat2, lng1, lng2 = X[i][0], Y[i][0], X[i][1], Y[i][1]
        radLat1 = rad(lat1)
        radLat2 = rad(lat2)
        a = radLat1 - radLat2
        b = rad(lng1) - rad(lng2)
        s = 2 * math.sin(math.sqrt(math.pow(math.sin(a / 2), 2)
                                   + math.cos(radLat1) * math.cos(radLat2)
                                   * math.pow(math.sin(b / 2), 2)))
        s = s * EARTH_RADIUS
        s = round(s * 10000) / 10000
        S = S + s * 1000
    S = S / (i + 1)
    return S
