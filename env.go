package cuberl

import (
	"errors"

	"github.com/unixpickle/gocube"
)

// Env is an anyrl.Env for cube states.
type Env struct {
	Objective Objective
	EpLen     int

	// FullState, if true, indicates that the maximum
	// number of solved pieces should be included in
	// observation vectors.
	FullState bool

	state    *State
	timestep int
}

// Reset generates a new state.
func (e *Env) Reset() ([]float64, error) {
	e.state = RandomStates(e.Objective, 1)[0]
	e.timestep = 0
	return e.vec(), nil
}

// Step takes a step in the environment.
//
// The input should be a one-hot move vector.
func (e *Env) Step(action []float64) (obs []float64, rew float64,
	done bool, err error) {
	if e.state == nil {
		err = errors.New("step Env: not initialized")
		return
	}
	for move, x := range action {
		if x != 0 {
			e.state, rew = e.state.Move(gocube.Move(move))
			break
		}
	}
	e.timestep++
	done = e.timestep == e.EpLen
	obs = e.vec()
	return
}

func (e *Env) vec() []float64 {
	res := CubeVector(&e.state.Cube)
	if e.FullState {
		res = append(res, float64(e.state.MaxSolved))
	}
	return res
}
