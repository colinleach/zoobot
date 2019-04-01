import pytest

import os

import production_service


@pytest.fixture(params=[False, True])
def has_previous_iterations(request):
    return request.param


@pytest.fixture(params=[False, True])
def has_old_instructions(request):
    return request.param


@pytest.fixture()
def base_dir(tmpdir):
    return tmpdir.mkdir('base_dir').strpath


@pytest.fixture()
def current_active_learning_state(base_dir, has_old_instructions, has_previous_iterations):
    if has_old_instructions:
        os.mkdir(os.path.join(base_dir, 'instructions_1111'))  # should ignore this
    os.mkdir(os.path.join(base_dir, 'instructions_1112'))  # should call this
    if has_previous_iterations:
        os.mkdir(os.path.join(base_dir, 'iteration_1111'))  # should ignore this
        os.mkdir(os.path.join(base_dir, 'iteration_1112'))  # should use this


def test_iterate(mocker, base_dir, current_active_learning_state, has_previous_iterations):
    mocker.patch('production_service.run_iteration.main', autospec=True)
    def mock_time():
        return 1234.87
    mocker.patch('production_service.time.time', new_callable=lambda: mock_time)
    production_service.iterate(base_dir, simulation=False, test=False)

    assert production_service.run_iteration.main.call_count == 1
    observed_call_args = production_service.run_iteration.main.call_args[1]  # named args 
    assert observed_call_args['instructions_dir'] == os.path.join(base_dir, 'instructions_1112')
    if has_previous_iterations:
        assert observed_call_args['previous_iteration_dir'] == os.path.join(base_dir, 'iteration_1112')
    else:
        assert observed_call_args['previous_iteration_dir'] is None

    assert observed_call_args['this_iteration_dir'] == 'iteration_1234.87'
    assert not observed_call_args['test'] 
