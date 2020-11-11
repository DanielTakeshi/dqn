from dqn.utils.io import read_json
import json
from os.path import join
import jsonmerge
from dqn.utils.io import create_output_dir
import random
import datetime
from dqn import common_config as CFG


_type_conversion_table = {
    "None": lambda _: None,
    "float": lambda x: float(x),
    "int": lambda x: int(x),
    "str": lambda x: str(x),
    "bool": lambda x: x.lower() == "true",
    "list": lambda x: list(x)
}


def convert_str_to_type(value, type_str):
    if type_str in _type_conversion_table:
        return _type_conversion_table[type_str](value)
    else:
        return None


class Configurations:

    def __init__(self, params, note):
        """D: establish of configuration-related and logging directory stuff.
        Main changes from Allen's code: save in hard disk rather than precious
        SSD space, added date string and remove annoying parentheses.
        """
        self.global_config = read_json(dir_path=params.exp_name,
                                       file_name="global_settings")
        self.exp_config = read_json(dir_path=params.exp_name, file_name=params.exp_id)
        self.exp_name = "_{}".format(params.exp_id)
        self.profile = params.profile

        if "seed" not in self.exp_config:
            self.exp_config["seed"] = random.randint(10, 100000)

        self.note = note
        self.params = jsonmerge.merge(
            base=self.global_config,
            head=self.exp_config)

        self.params["exp"] = self.exp_name

        if len(note) > 0:
            note = "_" + note.replace(" ", "_").lower()

        # D: extra stuff for more scalable logging, also check if teaching.
        date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        last = "{}{}_{}_s{}".format(self.params["exp"], note, date,
                                    self.params["seed"])

        if 'teacher' in self.params:
            assert len(self.params['teacher']['models']) >= 1
            head = CFG.SNAPS_STUDENT
        else:
            head = CFG.SNAPS_TEACHER
        self.params["log"]["dir"] = join(head, last)
        # __import__('pdb').set_trace()
        create_output_dir(params=self.params)

    def merge_keys(self, merge_key):
        self.params[merge_key] = {}
        for section_key, section in self.params.items():
            if merge_key in section:
                self.params[merge_key] = {
                    **self.params[merge_key],
                    **section[merge_key]}

    def dump(self, filename="params.txt"):
        with open(join(self.params["log"]["dir"], filename), 'w') as f:
            json.dump(self.params, f, sort_keys=True, indent=4)
            f.write('\n')
        return self.params
