from abc import ABC

class Aspect(ABC):
    """
    An base class representing an aspect of a user query, characterized by a description.

    :param description (str): A textual description of the aspect.
    """
    def __init__(self, description: str):
        self.description = description
        self.new_description = description
    
    def set_new_description(self, new_description: str):
        self.new_description = new_description
    
    def get_original_description(self) -> str:
        return self.description

    def get_new_description(self) -> str:
        return self.new_description
    
class Activity(Aspect):
    """
    A class representing an activity aspect of a user query, characterized by a description and a type.

    :param description (str): A textual description of the aspect.
    """
    def __init__(self, description: str):
        super().__init__(description)
        

class Broad(Aspect):
    """
    A class represents a soft preference in a user query.

    :param description (str): Initial description of the preference.
    """
    def __init__(self, description: str):
        super().__init__(description)
        
        
