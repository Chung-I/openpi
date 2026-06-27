from openpi.training.memory_labels import MemoryLabelConfig, MemoryLabelGenerator


def test_mock_backend_generates_labels():
    config = MemoryLabelConfig(backend="mock")
    generator = MemoryLabelGenerator(config)

    episodes = [
        {
            "goal": "clean the kitchen",
            "subtasks": [
                {"text": "pick up plate", "success": True, "timestamp": 0.0},
                {"text": "place plate in cabinet", "success": True, "timestamp": 5.0},
                {"text": "wipe counter", "success": False, "timestamp": 10.0},
                {"text": "wipe counter", "success": True, "timestamp": 15.0},
            ],
        }
    ]

    labels = generator.generate_labels(episodes)
    assert len(labels) == 1
    episode_labels = labels[0]
    # Should have one memory string per subtask
    assert len(episode_labels) == 4
    # Each label should be a string
    assert all(isinstance(m, str) for m in episode_labels)


def test_config_defaults():
    config = MemoryLabelConfig()
    assert config.backend == "claude"
    assert config.max_memory_tokens == 128
