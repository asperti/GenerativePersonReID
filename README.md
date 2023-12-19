# GenerativePersonReID
**A generative approach to person re-identification**

Person Re-identification is the problem of identifying comparable subjects across a network of nonoverlapping cameras. Typically, this is accomplished by deriving a vector of distinctive features from the source image that represent the specific person captured by the camera. Developing a robust, invariant, and discriminative set of features is a challenging undertaking, often utilizing contrastive learning.

In this work, we explores an alternative approach, where the representation of an individual is learned as the conditioning information needed to generate images of that specific person in different poses and backgounds. By doing so, we untether the identity of the individual from other information related to a specific instance, captured in the noise. This decoupling allows for intriguing explorations of the underlying latent spaces, and interesting transformations from one identity to another. The generative models employed in this research are recent diffusion models, known for their sensitivity to conditioning across various contexts.

-----

The first step consists in training a generative model of persons, conditioned over the identity of the person.

[]()

