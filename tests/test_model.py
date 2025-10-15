from model.data_preparing_and_visualization import X, y

def test_prepare_data():
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 1
    assert len(y.shape) == 1