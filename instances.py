class Instances:
    def __init__(self, image_size, **kwargs):
        self._image_size = image_size
        self._fields = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self):
        return self._image_size

    def __setattr__(self, name, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name):
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name, value):
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name):
        return name in self._fields

    def remove(self, name):
        del self._fields[name]

    def get(self, name):
        return self._fields[name]

    def get_fields(self):
        return self._fields

    def __getitem__(self, item):
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self):
        for v in self._fields.values():
            return len(v)
        return 0

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    def __str__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, v)
        s += "])"
        return s
