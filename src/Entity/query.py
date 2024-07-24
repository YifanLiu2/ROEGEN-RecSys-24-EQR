from .aspect import *

class Query:
    """
    A class represents a user's travel destinations query.
    A query contains a list of broad and constraints.

    :param description (str): A brief description of the query.
    :param broad (list[Broad]): A list of broad aspect.
    :param activites (list[Activity]): A list of activities.
    """
    def __init__(self, description: str, broad=None, activities=None):
        self.description = description
        self.broad = broad if broad is not None else []
        self.activities = activities if activities is not None else []

    def get_all_aspects(self) -> list[Aspect]:
        """
        Return a list of aspects in user query.
        """
        # otherwise, return all aspects
        return self.get_broad() + self.get_activities()
        
    def get_description(self) -> str:
        """
        Return the description of the query.
        """
        return self.description
    
    def get_broad(self) -> list[Broad]:
        """
        Return a list of user broad.
        """
        return self.broad
        
    def get_activities(self) -> list[Activity]:
        """
        Return a list of activities.
        """
        return self.activities
    
    def add_broad(self, broad: Broad):
        """
        Adds a broad to the query.

        :param broad (broad): The broad to be added.
        """
        self.broad.append(broad)
    
    def add_activity(self, activity: Activity):
        """
        Adds a activity aspect to the query.

        :param activity (activity): The activitiy to be added.
        """
        self.activities.append(activity)