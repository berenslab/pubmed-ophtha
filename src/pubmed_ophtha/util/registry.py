"""Class Registry for managing model classes."""


class Registry:
    """
    A registry for managing and storing model classes.

    This class provides methods to register, retrieve, and manage model classes by name.
    """

    def __init__(self):
        """Initialize the registry to store classes."""
        self._registry = {}

    def __contains__(self, name: str):
        """
        Check if a class is registered in the registry.

        Args:
            name (str): Name of the class to check.

        Returns:
            bool: True if the class is registered, False otherwise.

        """
        return name in self._registry

    def __len__(self):
        """
        Get the number of classes registered in the registry.

        Returns:
            int: Number of registered classes.

        """
        return len(self._registry)

    def get(self, name: str):
        """
        Get a class from the registry by its name.

        Args:
            name (str): Name of the class to retrieve.

        Returns:
            type: The class associated with the given name.

        Raises:
            KeyError: If the name is not found in the registry.

        """
        if name not in self._registry:
            raise KeyError(f"'{name}' is not registered.")
        return self._registry[name]

    def get_by_model_name(self, model_name: str):
        """
        Get a class from the registry by its model name.

        Args:
            model_name (str): Model name to retrieve the class for.

        Returns:
            type: The class associated with the given model name.

        Raises:
            KeyError: If the model name is not found in the registry.

        """
        for cls in self._registry.values():
            if (
                hasattr(cls, "SUPPORTED_MODEL_NAMES")
                and model_name in cls.SUPPORTED_MODEL_NAMES
            ):
                return cls
        raise KeyError(f"Model name '{model_name}' is not registered.")

    def register(self, name: str | None = None):
        """
        Register a class in the registry.

        Args:
            name (str, optional): Name of the model. If None, the class name is used.

        """

        def decorator(cls):
            cls_name = name or cls.__name__
            if cls_name in self._registry:
                raise ValueError(f"'{cls_name}' is already registered.")
            self._registry[cls_name] = cls
            return cls

        return decorator

    def __iter__(self):
        """
        Iterate over the registered classes.

        Yields:
            tuple: A tuple containing the class name and the class itself.

        """
        yield from self._registry.items()
