from openpi.training.memory_labels import Episode, MemoryLabelConfig, MemoryLabelGenerator, MemoryLabels


def test_mock_backend_generates_labels():
    config = MemoryLabelConfig(backend="mock")
    generator = MemoryLabelGenerator(config)

    episodes = [
        Episode(
            goal="clean the kitchen",
            subtasks=[
                "pick up plate",
                "place plate in cabinet",
                "wipe counter",
                "wipe counter",
            ],
            success_flags=[True, True, False, True],
        )
    ]

    labels = generator.generate_labels(episodes)
    assert len(labels) == 1
    result = labels[0]
    assert isinstance(result, MemoryLabels)
    assert result.episode_id == "0"
    # Should have one memory string per subtask
    assert len(result.memories) == 4
    # Each label should be a string
    assert all(isinstance(m, str) for m in result.memories)


def test_config_defaults():
    config = MemoryLabelConfig()
    assert config.backend == "claude"
    assert config.max_memory_tokens == 128


def test_resolved_api_key_env_defaults():
    claude_config = MemoryLabelConfig(backend="claude")
    assert claude_config.resolved_api_key_env == "ANTHROPIC_API_KEY"

    openai_config = MemoryLabelConfig(backend="openai")
    assert openai_config.resolved_api_key_env == "OPENAI_API_KEY"

    explicit_config = MemoryLabelConfig(backend="openai", api_key_env="MY_CUSTOM_KEY")
    assert explicit_config.resolved_api_key_env == "MY_CUSTOM_KEY"
