use rs_math::stats::dist;

#[test]
fn test_uniform_he() {
    let nsamples = 10000;
    let uni_dist = dist::uniform_he(nsamples, 20).unwrap();

    // Test length
    assert!(
        uni_dist.len() == nsamples,
        "Length of the sample distrubution is invalid [ EXPECTED: {} | FOUND: {} ]",
        nsamples,
        uni_dist.len()
    );

    // Test for zero mean: Length >= 10000 (Law of large numbers)
    let mean = uni_dist.iter().sum::<f32>() / uni_dist.len() as f32;
    assert!(
        mean < 0.01,
        "The mean of the sample distribution is not zero [ MEAN: {} ]",
        mean
    );

    // TODO: Check for variance
}

#[test]
fn test_normal_he() {
    let nsamples = 10000;
    let norm_dist = dist::normal_he(nsamples, 20).unwrap();

    // Test length
    assert!(
        norm_dist.len() == nsamples,
        "Length of the sample distrubution is invalid [ EXPECTED: {} | FOUND: {} ]",
        nsamples,
        norm_dist.len()
    );

    // Test for zero mean: Length >= 10000 (Law of large numbers)
    let mean = norm_dist.iter().sum::<f32>() / norm_dist.len() as f32;
    assert!(
        mean < 0.01,
        "The mean of the sample distribution is not zero [ MEAN: {} ]",
        mean
    );

    // TODO: Check for variance

    // TODO: Check for standard deviation (from variance)
}

