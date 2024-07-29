def pytest_addoption(parser):
    parser.addoption("--dtype", type=str)
    parser.addoption("--model", type=str)


def pytest_generate_tests(metafunc):
    if "dtype" in metafunc.fixturenames:
        if metafunc.config.getoption("--dtype") is not None:
            dtypes = metafunc.config.getoption("--dtype").split(',')
        else:
            dtypes = ["f32", "q2_k", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
        metafunc.parametrize("dtype", dtypes)
    if "model" in metafunc.fixturenames:
        if metafunc.config.getoption("--model") is not None:
            models = metafunc.config.getoption("--model").split(',')
        else:
            models = ["TinyLlama_v1.1"]
        metafunc.parametrize("model", models)