from research.rangehunter_m1_trendfollow_v5_staged_503020_nautilus_raw_bt import P5, main

# Ablation: preserve V4 signal and favorable-only staged 50/30/20,
# but disable the over-aggressive early fast-fail observed in V5.
P5['fast_fail_r'] = -999.0
P5['fast_fail_sec'] = 0

if __name__ == '__main__':
    main()
