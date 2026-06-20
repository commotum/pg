### Quaternion Units as Basis Elements and Rotors

Quaternion notation can be confusing because $i$, $j$, and $k$ have two related but distinct interpretations. Within the four-dimensional quaternion algebra, they are basis elements satisfying

$$
i^2=j^2=k^2=ijk=-1.
$$

Left-multiplication by $i$ acts like a $90^\circ$ turn in quaternion space: $1\mapsto i\mapsto-1$ and $j\mapsto k\mapsto-j$. This makes identities such as $ij=k$ and $ijk=-1$ geometrically natural.

When quaternions are used to rotate ordinary three-dimensional vectors, however, the rotation is performed by conjugation:

$$
v' = qvq^{-1}.
$$

A rotation through angle $\theta$ about the unit axis $u$ is represented by

$$
q=\cos(\theta/2)+u\sin(\theta/2).
$$

Consequently, $i$, $j$, and $k$ each represent a $180^\circ$ spatial rotation about their respective axes. Graphics APIs often conceal the full conjugation operation behind notation such as $qv$, making it easy to mistake this use for ordinary quaternion multiplication.

Thus $i$ can describe a $90^\circ$ turn under multiplication within quaternion space while also representing a $180^\circ$ rotation when acting on three-dimensional vectors. The apparent contradiction comes from confusing these two different actions.