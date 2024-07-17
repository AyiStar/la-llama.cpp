def pytest_addoption(parser):
    parser.addoption("--dtype", type=str)


def pytest_generate_tests(metafunc):
    if "dtype" in metafunc.fixturenames:
        if metafunc.config.getoption("--dtype") is not None:
            dtypes = [metafunc.config.getoption("--dtype")]
        else:
            dtypes = ["f32", "q4_0", "q4_1", "q8_0", "q5_0"]
        metafunc.parametrize("dtype", dtypes)