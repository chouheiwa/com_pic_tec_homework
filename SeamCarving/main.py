import SeamCarving.seam_carving as sc

if __name__ == '__main__':
    simple = sc.SeamCarvingGradientMagnitude('images/test.png')
    simple.test_job()
    simple.normal_job(300, 426)
