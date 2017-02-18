package cuberl

import (
	"math"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/gocube"
)

// State implements cube dynamics and a reward mechanism.
type State struct {
	// Cube is the current state of the cube.
	Cube gocube.CubieCube

	// MaxSolved is used to keep track of the maximum number
	// of pieces that were solved simultaneously in any
	// timestep since the start of the episode.
	// start of the episode.
	MaxSolved int
}

// Move produces a new State and an immediate reward
// from applying the move m to the state e.
func (s *State) Move(m gocube.Move) (*State, float64) {
	newS := *s
	newS.Cube.Move(m)
	newS.MaxSolved = essentials.MaxInt(newS.MaxSolved, newS.NumSolved())
	return &newS, math.Max(0, float64(newS.NumSolved()-s.MaxSolved))
}

// NumSolved returns the number of solved pieces.
// This will range from 0 to 20.
func (s *State) NumSolved() int {
	var c int
	for i, x := range s.Cube.Edges[:] {
		if x.Piece == i && !x.Flip {
			c++
		}
	}
	for i, x := range s.Cube.Corners[:] {
		if x.Piece == i && x.Orientation == 1 {
			c++
		}
	}
	return c
}

// CubeVector produces a vector representation of the
// cube state.
func CubeVector(creator anyvec.Creator, cube *gocube.CubieCube) anyvec.Vector {
	stickers := cube.StickerCube()
	data := make([]float64, 0, 8*6*6)
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
