def get_f(eq_focal=26, w_d_ratio=3/5, video_width=1080, pxs_in_width=3000, eff_px_in_width=2160):
    return int(video_width*eq_focal*pxs_in_width / (43.23*w_d_ratio*eff_px_in_width))


print(get_f(26, 3/5, 1080, 3000, 2160))