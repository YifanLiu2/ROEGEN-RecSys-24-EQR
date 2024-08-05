class AbstractQuery:
    """
    A class represents a user's travel destinations query.
    A query contains a list of broad and constraints.

    :param description (str): A brief description of the query.
    :param broad (list[Broad]): A list of broad aspect.
    :param activites (list[Activity]): A list of activities.
    """
    def __init__(self, description: str):
        self.description = description
        self.reformulation = None
    

    def get_description(self) -> str:
        """
        Return the description of the query.
        """
        return self.description
    

    def set_reformuation(self, reformulation: str):
        self.reformulation = reformulation
    

    def get_reformulation(self) -> str:
        """
        """
        if self.reformulation:
            return self.reformulation
        else:
            return self.description
    

class Activity(AbstractQuery):
    """
    """
    def __init__(self, description: str):
        super().__init__(description)
        

class Broad(AbstractQuery):
    """
    """
    def __init__(self, description: str):
        super().__init__(description)
