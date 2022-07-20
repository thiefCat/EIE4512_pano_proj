def get_f(eq_focal=26, w_d_ratio=3/5, video_width=1080, pxs_in_width=3000, eff_px_in_width=2160):
    '''
    eq_focal        : equivalent optical focal length of the camera to full frame sensor
    w_d_ratio       : width / digonal of the camera sensor
    video_width     : width of the video
    pxs_in_width    : total pixels in width
    eff_px_in_width : effective pixels in width for video taking (e.g., 2160px out of 3000px )
    - returns the digital focal length (pixels) for cylindrical view
    '''
    return int(video_width*eq_focal*pxs_in_width / (43.23*w_d_ratio*eff_px_in_width))


# print(get_f(26, 3/5, 1080, 3000, 2160))