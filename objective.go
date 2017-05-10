package cuberl

import (
	"errors"

	"github.com/unixpickle/gocube"
)

// ObjectiveUsage is a usage string for a command-line
// "-objective" flag.
const ObjectiveUsage = "agent objective (FullCube, FirstLayer, or PetrusBlock)"

// Objective indicates what part of the cube the agent is
// supposed to be solving.
type Objective int

const (
	FullCube Objective = iota
	FirstLayer
	PetrusBlock
)

// String returns the human-readable name for the
// Objective.
func (o *Objective) String() string {
	switch *o {
	case FullCube:
		return "FullCube"
	case FirstLayer:
		return "FirstLayer"
	case PetrusBlock:
		return "PetrusBlock"
	default:
		panic("invaild Objective")
	}
}

// Set sets the Objective from a human-readable name.
func (o *Objective) Set(val string) error {
	switch val {
	case "FullCube":
		*o = FullCube
	case "FirstLayer":
		*o = FirstLayer
	default:
		return errors.New("unknown objective")
	}
	return nil
}

// Evaluate counts the number of pieces solved under this
// objective.
func (o *Objective) Evaluate(cube *gocube.CubieCube) int {
	var c int
	switch *o {
	case FullCube:
		for i, x := range cube.Edges[:] {
			if x.Piece == i && !x.Flip {
				c++
			}
		}
		for i, x := range cube.Corners[:] {
			if x.Piece == i && x.Orientation == 1 {
				c++
			}
		}
	case FirstLayer, PetrusBlock:
		var edges, corners []int
		if *o == FirstLayer {
			edges = []int{2, 8, 10, 11}
			corners = []int{0, 1, 4, 5}
		} else if *o == PetrusBlock {
			edges = []int{2, 3, 8, 9, 10}
			corners = []int{0, 4}
		}
		for _, i := range edges {
			x := cube.Edges[i]
			if x.Piece == i && !x.Flip {
				c++
			}
		}
		for _, i := range corners {
			x := cube.Corners[i]
			if x.Piece == i && x.Orientation == 1 {
				c++
			}
		}
	default:
		panic("invalid Objective")
	}
	return c
}
