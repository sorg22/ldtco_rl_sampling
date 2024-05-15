from rl_sampling import UCBPL


if __name__ == "__main__":
    fname = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_10_2024/toSungwon/out.table"
    spec = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_10_2024/toSungwon/out.spec"
    model_file = "/nfs/site/disks/ad_wa_skim501/graphrp_test/golden_test_4_10_2024/model_best_doe.pt"
    ucb = UCBPL(fname, spec, model_file, 3000, "res_5000.csv")
