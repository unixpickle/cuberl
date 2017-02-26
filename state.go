package cuberl

import (
	"math"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/gocube"
)

// NumActions is the number of distinct Rubik's cube moves
// in the half-turn metric.
const NumActions = 18

// CubeVectorSize is the size of vectors produced by
// CubeVector.
const CubeVectorSize = 8 * 6 * 6

// State implements cube dynamics and a reward mechanism.
type State struct {
	Objective Objective

	// Cube is the current state of the cube.
	Cube gocube.CubieCube

	// MaxSolved is used to keep track of the maximum number
	// of pieces that were solved simultaneously in any
	// timestep since the start of the episode.
	// start of the episode.
	MaxSolved int
}

// RandomStates produces n random start states.
func RandomStates(o Objective, n int) []*State {
	res := make([]*State, n)
	for i := range res {
		res[i] = &State{Cube: gocube.RandomCubieCube(), Objective: o}
		res[i].MaxSolved = res[i].NumSolved()
	}
	return res
}

// Move produces a new State and an immediate reward
// from applying the move m to the state e.
func (s *State) Move(m gocube.Move) (*State, float64) {
	newS := *s
	newS.Cube.Move(m)
	newS.MaxSolved = essentials.MaxInt(newS.MaxSolved, newS.NumSolved())
	return &newS, math.Max(0, float64(newS.NumSolved()-s.MaxSolved))
}

// NumSolved returns the number of solved pieces according
// to the objective.
func (s *State) NumSolved() int {
	return s.Objective.Evaluate(&s.Cube)
}

// CubeVector produces a vector representation of the
// cube state.
func CubeVector(creator anyvec.Creator, cube *gocube.CubieCube) anyvec.Vector {
	stickers := cube.StickerCube()
	data := make([]float64, 0, CubeVectorSize)
	for i, x := range stickers[:] {
		if (i-4)%9 == 0 {
			// Avoid the center pieces.
			continue
		}
		subVec := make([]float64, 6)
		subVec[x-1] = 1
		data = append(data, subVec...)
	}
	return creator.MakeVectorData(creator.MakeNumericList(data))
}
