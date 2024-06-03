from abc import ABC, abstractmethod

class Constraint(ABC):
    def __init__(self, description, is_verifiable):
        self.description = description
        self.is_verifiable = is_verifiable

    def verifiable(self):
        return self.is_verifiable

    @abstractmethod
    def get_query(self):
        pass


class SpecificConstraint(Constraint):
    def __init__(self, description, is_verifiable):
        super().__init__(description, is_verifiable)
    
    def get_query(self):
        return [self.description]


class GeneralConstraint(Constraint):
    def __init__(self, description, is_verifiable):
        super().__init__(description, is_verifiable)
        self.expansion = [self.description]

    def get_query(self):
        return self.expansion

    def expand(self, new_description):
        self.expansion.append(new_description)


class Query:
    def __init__(self, description: str):
        self.description= description
        self.constraints = []

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
