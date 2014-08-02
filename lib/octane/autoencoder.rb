module Octane
	class AutoEncoder
		attr_reader :encoder, :decoder
		def initialize(input_size, code_size, tied_weights=false, options={})
			@encoder=Network.new(options[:learning_rate], options[:weight_decay])
			@encoder.add_layer input_size
			@encoder.add_layer code_size, options[:code_layer_type]

			@decoder=Network.new(options[:learning_rate], options[:weight_decay])
			@decoder.add_layer code_size
			@decoder.add_layer input_size, options[:output_layer_type]
			
		end

		def encode(input)
			@encoder.forward_pass(input)
			@encoder.hidden_layer.map(&:last_squashed)
		end

		def decode(code)
			@decoder.forward_pass(code)
			@decoder.hidden_layer.map(&:last_squashed)
		end

		def train_one(input, output=nil)
			output=input unless output
			train_decoder(input, output)
			train_encoder(input, output)
		end

		def train_decoder(input, output)
			code=encode(input)
			@decoder.train_one([code, output])
		end

		def train_encoder(input, output)
			# assumes that train_decoder was called first
			@encoder.output_layer.zip(@decoder.input_layer.map &:delta).map {|neuron, delta| neuron.delta=delta}
			@encoder.propagate_deltas_back
			@encoder.calculate_weight_changes
			@encoder.apply_weight_changes 1
		end

		def test(dataset)
			err=0
			error=->(input, reconstruction){input.zip(reconstruction).map {|i, r| (i-r)**2}.reduce(:+)/(2*input.length)}
			dataset.data.each {|example, output|
				reconstruction=decode(encode(example))
				err+=error[reconstruction, output]
			}
			err/(2*dataset.size)
		end
	end
end