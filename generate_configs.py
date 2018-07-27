import json
from copy import deepcopy
from random import randint, seed


def write_json(filename, dictionary):
    with open(filename, 'w') as data_file:
        json.dump(dictionary, data_file, sort_keys=True,indent=4)

def read_json(filename):
    with open(filename) as data_file:
        config = json.load(data_file)
    return config

def run():
    seed(7)
    default_config = read_json("default_config.json")

    # Default
    newconfig = deepcopy(default_config)
    newconfig["root_name"]="default_0"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)

    # Scale randomness
    newconfig = deepcopy(default_config)
    newconfig["scale_randomness"]=1
    newconfig["root_name"]="scale_randomness_1"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)
    newconfig["scale_randomness"]=2
    newconfig["root_name"]="scale_randomness_2"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)

    # Translation
    newconfig = deepcopy(default_config)
    newconfig["translation"]=2
    newconfig["root_name"]="translation_2"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)

    # Rotation
    newconfig = deepcopy(default_config)
    newconfig["rotation"]=1
    newconfig["root_name"]="rotation_01"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)

    # Noise
    newconfig = deepcopy(default_config)
    newconfig["noise"]=1
    newconfig["root_name"]="noise_1"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)

    # Transform/Warp
    newconfig = deepcopy(default_config)
    newconfig["transformation"]=0.1
    newconfig["root_name"]="transformation_01"
    newconfig["seed"]=randint(0, 10000)
    write_json("./configs/"+newconfig["root_name"]+".json", newconfig)

    # scale_randomness = 0
    # sampling_rate = 1000
    # noise = 0
    # # Adjust Translations, rotations, transformations
    # for translation in [0,2,4]:
    #     for rotation in [0,1]:
    #         for transformation in [0,.1,.2]:
    #             updates = {"translation":translation,
    #                        "rotation":rotation,
    #                        "scale_randomness":scale_randomness,
    #                        "noise":noise,
    #                        "transformation":transformation,
    #                        "sampling_rate":sampling_rate,
    #                        "root_name":"sr{0}_trs{1}_trf{2}_rot{3}_sclr{4}_no{5}".format(sampling_rate,translation,transformation,rotation,scale_randomness,noise)}
    #             default_config.update(updates)
    #             write_json("./configs/"+updates["root_name"]+"_config.json",default_config)
    # default_config = read_json("default_config.json")
    # # Adjust
    # translation = 2
    # rotation = 1
    # transformation = .1
    # for sampling_rate in [1000]:
    #     for scale_randomness in [0,1,3]:
    #         for noise in [0,1,2]:
    #             updates = {"translation":translation,
    #                        "rotation":rotation,
    #                        "scale_randomness":scale_randomness,
    #                        "noise":noise,
    #                        "transformation":transformation,
    #                        "sampling_rate":sampling_rate,
    #                        "root_name":"sr{0}_trs{1}_trf{2}_rot{3}_sclr{4}_no{5}".format(sampling_rate,translation,transformation,rotation,scale_randomness,noise)}
    #             default_config.update(updates)
    #             write_json("./configs/"+updates["root_name"]+"_config.json",default_config)


def run1():
    default_config = read_json("default_config.json")
    scale_randomness = 0
    sampling_rate = 10000
    noise = 0

    #default
    write_json("./configs/novar_config.json",default_config)

    # adjust translation
    for translation in [2,4]:
      updates = {"translation": translation,
                 "root_name":"trs{0}".format(translation)}
      default_config.update(updates)
      write_json("./configs/"+updates["root_name"]+"_config.json",default_config)
    # adjust rotation
    for rotation in [1]:
      updates = {"rotation": rotation,
                 "root_name":"rot{0}".format(rotation)}
      default_config.update(updates)
      write_json("./configs/"+updates["root_name"]+"_config.json",default_config)
    # adjust warp
    for warp in [0.1,0.2]:
      updates = {"transformation": warp,
                 "root_name":"warp{0}".format(warp)}
      default_config.update(updates)
      write_json("./configs/"+updates["root_name"]+"_config.json",default_config)

    # adjust scale randomness
    for sr in [1,3]:
      updates = {"scale_randomness": sr,
                 "root_name":"sr{0}".format(sr)}
      default_config.update(updates)
      write_json("./configs/"+updates["root_name"]+"_config.json",default_config)

    # adjust learning rate
    for no in [1,2]:
      updates = {"noise": no,
                 "root_name":"no{0}".format(no)}
      default_config.update(updates)
      write_json("./configs/"+updates["root_name"]+"_config.json",default_config)

def run2():
  default_config = read_json("default_config.json")

  for convlayers in [1,2]:
    for sampling in [100, 1000, 10000]:
      # vary scale randomness
      newconfig = deepcopy(default_config)
      updates = {"scale_randomness": 3,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_sr5".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # vary rotation
      newconfig = deepcopy(default_config)
      updates = {"rotation": 1,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_rot1".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # vary translation
      newconfig = deepcopy(default_config)
      updates = {"translation": 4,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_trs4".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # vary transformation
      newconfig = deepcopy(default_config)
      updates = {"transformation": 3,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_trf3".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # vary poly_transformation
      newconfig = deepcopy(default_config)
      updates = {"poly_transformation": 5,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_ptrf3".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # vary noise
      newconfig = deepcopy(default_config)
      updates = {"noise": 1,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_no1".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # all default
      newconfig = deepcopy(default_config)
      updates = { "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_novar".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # all on (except for polytransform)
      newconfig = deepcopy(default_config)
      updates = {"noise": 1,
                  "scale_randomness": 3,
                  "rotation": 1,
                  "translation": 4,
                  "transformation": 3,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_AllOnButPoly".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

      # all on (including polytransform)
      newconfig = deepcopy(default_config)
      updates = {"noise": 1,
                  "scale_randomness": 3,
                  "rotation": 1,
                  "translation": 4,
                  "transformation": 3,
                  "poly_transformation": 5,
                  "sampling_rate": sampling,
                  "layer_count": convlayers,
                  "root_name": "convlayers{0}_sampling{1}_AllOnWithPoly".format(convlayers,sampling)}
      newconfig.update(updates)
      newconfig["random_seed"]=randint(0, 10000)
      write_json("./configs/"+updates["root_name"]+"_config.json",newconfig)

if __name__ == '__main__':
    """
    The main function called when main.py is run
    from the command line:

    > python main.py
    """
    # uncomment to run mixed-parameter generation
    #run()

    # uncomment to run single-parameter variance generation
    #run1()

    run2()
