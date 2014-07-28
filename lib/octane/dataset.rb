module Octane
	class Dataset
		include Enumerable

		def initialize
			@dataset=[]
		end

		def each
			@dataset.each {|e| yield e}
		end

		def add_sample(sample,expected)
			@dataset<<[sample,expected.is_a?(Array) ? expected : [expected]]
		end

		def data
			@dataset
		end

		def batch(size)
			@dataset.sample(size)
		end

		def data=(data)
			@dataset=data
		end

		def clear
			@dataset=[]
		end

		def clone
			new_dataset=Dataset.new
			new_dataset.data=Marshal.load(Marshal.dump(@dataset.clone)) # TODO: god that's ugly.
			new_dataset
		end

		def size
			@dataset.size
		end

		def sample(sz=1)
			if sz==1
				@dataset.sample
			else
				@dataset.sample sz
			end
		end
	end
end