#!/usr/bin/python
import json
import sys

# @author Wellington Castro <wvmcastro>

class JsonHandler:
    """ This class is made to both write and read from json files easier """

    def __init__(self):
        pass

    def read(self, file_name):
        """ Takes a json file name and return a dict object with its content """
        dictionary = dict()
        try:
            params_file = open(file_name, "r")
            dictionary = json.loads(params_file.read())
            params_file.close()
        except:
            e = sys.exc_info()[0]
            print "Error: ", e

        return dictionary


    def write(self, dictionary, file_name):
        """ Writes the dictionary content in a json file """
        sucess = True

        try:
            file = open(file_name, "w+")
            json.dump(dictionary, file)
            file.close()
        except:
            e = sys.exc_info()[0]
            print "Erro:", e
            sucess = False

        return sucess
